// KCF tracker C++ 主进程 + Python GDino 子进程 (pipe + 共享内存)
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <mutex>
#include <atomic>

// 配置
const char* get_env(const char* key, const char* def) {
    const char* v = std::getenv(key);
    return v ? v : def;
}
const char* VIDEO = get_env("VIDEO", "input.mp4");
const char* QUERY = get_env("QUERY", "object");
const char* OUT   = get_env("OUTPUT", "output.mp4");
const char* PYTHON = get_env("PYTHON", "python3");
const float CONF_THRESHOLD = 0.5f;
const int MAX_MISS = 2;
const int TARGET_FPS = 30;
const float FRAME_INTERVAL = 1.0f / TARGET_FPS;
const float TRACK_SCALE = 1.0f / 3.0f;
int SHM_H = 0, SHM_W = 0;
const int SHM_C = 3;
int SHM_SIZE = 0;
const char* SHM_NAME = "/kcf_gdino_shm";

// 动量预测
float vel_x = 0, vel_y = 0;
float acc_x = 0, acc_y = 0;
float prev_vx = 0, prev_vy = 0;
cv::Rect prev_rect;
bool has_prev = false;
const float VEL_ALPHA = 0.6f;
const float ACC_ALPHA = 0.4f;

void update_momentum(const cv::Rect& cur) {
    if (has_prev) {
        float raw_vx = cur.x - prev_rect.x;
        float raw_vy = cur.y - prev_rect.y;
        vel_x = VEL_ALPHA * raw_vx + (1-VEL_ALPHA) * vel_x;
        vel_y = VEL_ALPHA * raw_vy + (1-VEL_ALPHA) * vel_y;
        float raw_ax = vel_x - prev_vx;
        float raw_ay = vel_y - prev_vy;
        acc_x = ACC_ALPHA * raw_ax + (1-ACC_ALPHA) * acc_x;
        acc_y = ACC_ALPHA * raw_ay + (1-ACC_ALPHA) * acc_y;
        prev_vx = vel_x;
        prev_vy = vel_y;
    }
    prev_rect = cur;
    has_prev = true;
}

cv::Rect predict_next() {
    int px = prev_rect.x + int(vel_x + 0.5f * acc_x);
    int py = prev_rect.y + int(vel_y + 0.5f * acc_y);
    return cv::Rect(px, py, prev_rect.width, prev_rect.height);
}

// 全局状态
std::mutex result_mtx;
struct GDinoResult {
    bool valid = false;
    cv::Rect bbox;
    float score = 0;
    int frame_idx = 0;
    float det_ms = 0;
};
GDinoResult latest_result;
std::atomic<bool> gdino_pending(false);

// 共享内存
uint8_t* shm_ptr = nullptr;
int shm_fd = -1;

bool setup_shm() {
    shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd < 0) { perror("shm_open"); return false; }
    ftruncate(shm_fd, SHM_SIZE);
    shm_ptr = (uint8_t*)mmap(0, SHM_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
    return shm_ptr != MAP_FAILED;
}

void cleanup_shm() {
    if (shm_ptr) munmap(shm_ptr, SHM_SIZE);
    if (shm_fd >= 0) close(shm_fd);
    shm_unlink(SHM_NAME);
}

void write_shm(const cv::Mat& frame) {
    memcpy(shm_ptr, frame.data, SHM_SIZE);
}

// Pipe 通信
FILE* to_gdino = nullptr;
FILE* from_gdino = nullptr;

void send_detect(const cv::Mat& frame, int idx, int orig_h, int orig_w) {
    write_shm(frame);
    fprintf(to_gdino, "DETECT %d %d %d %s\n", idx, orig_h, orig_w, QUERY);
    fflush(to_gdino);
    gdino_pending = true;
}

bool poll_result(GDinoResult& out) {
    int fd = fileno(from_gdino);
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(fd, &fds);
    struct timeval tv = {0, 0};
    if (select(fd+1, &fds, NULL, NULL, &tv) <= 0) return false;

    char line[512];
    if (!fgets(line, sizeof(line), from_gdino)) return false;

    int fidx;
    char bbox_str[64];
    float score, dt;
    sscanf(line, "RESULT %d %s %f %f", &fidx, bbox_str, &score, &dt);

    out.frame_idx = fidx;
    out.score = score;
    out.det_ms = dt;
    gdino_pending = false;

    if (strcmp(bbox_str, "NONE") == 0) {
        out.valid = false;
        return true;
    }
    int x, y, w, h;
    sscanf(bbox_str, "%d,%d,%d,%d", &x, &y, &w, &h);
    out.bbox = cv::Rect(x, y, w, h);
    out.valid = true;
    return true;
}

cv::Rect scale_bbox(const cv::Rect& r, float s) {
    return cv::Rect(int(r.x*s), int(r.y*s), int(r.width*s), int(r.height*s));
}
cv::Rect scale_bbox(const cv::Rect2d& r, float s) {
    return cv::Rect(int(r.x*s), int(r.y*s), int(r.width*s), int(r.height*s));
}

// 追踪逻辑（file/live 共用）：处理一帧的 GDino 结果 + KCF 更新
// 返回当前帧的 tag
std::string process_frame(const cv::Mat& frame, int i, int h, int w, int sw, int sh,
                          std::map<int, cv::Mat>& frame_buffer,
                          cv::Ptr<cv::TrackerKCF>& tracker,
                          bool& tracking, cv::Rect& rect, int& miss_count,
                          int& corrections, int& confirms, int& det_count) {
    std::string tag = "";
    bool corrected = false;

    GDinoResult res;
    if (poll_result(res)) {
        det_count++;
        if (res.valid && res.score >= CONF_THRESHOLD) {
            bool need_reinit = true;
            if (tracking && rect.width > 0) {
                float cur_cx = rect.x + rect.width/2.0f;
                float cur_cy = rect.y + rect.height/2.0f;
                float det_cx = res.bbox.x + res.bbox.width/2.0f;
                float det_cy = res.bbox.y + res.bbox.height/2.0f;
                float dist = std::sqrt(std::pow(cur_cx-det_cx,2) + std::pow(cur_cy-det_cy,2));
                float obj_size = std::max(rect.width, rect.height);
                if (dist < obj_size * 0.8f) {
                    need_reinit = false;
                    confirms++;
                    tag = "CONFIRM";
                }
            }
            if (need_reinit) {
                cv::Mat init_frame = frame_buffer.count(res.frame_idx) ?
                    frame_buffer[res.frame_idx] : frame;
                cv::Mat small;
                cv::resize(init_frame, small, cv::Size(sw, sh));
                tracker = cv::TrackerKCF::create();
                tracker->init(small, scale_bbox(res.bbox, TRACK_SCALE));
                rect = res.bbox;
                for (int fi = res.frame_idx+1; fi < std::min(i, res.frame_idx+11); fi++) {
                    if (frame_buffer.count(fi)) {
                        cv::Mat s; cv::resize(frame_buffer[fi], s, cv::Size(sw, sh));
                        cv::Rect2d r;
                        if (tracker->update(s, r))
                            rect = scale_bbox(r, 1.0f/TRACK_SCALE);
                    }
                }
                cv::Mat s; cv::resize(frame, s, cv::Size(sw, sh));
                cv::Rect2d r;
                if (tracker->update(s, r))
                    rect = scale_bbox(r, 1.0f/TRACK_SCALE);
                tag = "CORR";
            }
            tracking = true;
            miss_count = 0;
            corrections++;
            corrected = true;
        } else if (res.valid) {
            miss_count++;
            tag = "LOW";
            if (miss_count >= MAX_MISS) { tracking = false; tag = "GONE"; }
        } else {
            miss_count++;
            tag = "NO-DET";
            if (miss_count >= MAX_MISS) { tracking = false; tag = "GONE"; }
        }
        if (!gdino_pending) send_detect(frame, i, h, w);
    }

    if (tracking && tracker && !corrected) {
        cv::Mat small;
        cv::resize(frame, small, cv::Size(sw, sh));
        cv::Rect2d r;
        if (tracker->update(small, r))
            rect = scale_bbox(r, 1.0f/TRACK_SCALE);
        else
            tracking = false;
    }

    if (!gdino_pending && tracking) send_detect(frame, i, h, w);

    if (tracking && rect.width > 0) {
        if (tag.empty()) tag = "TRACK";
        update_momentum(rect);
    } else {
        if (tag.empty()) tag = "WAIT";
    }

    return tag;
}

int main() {
    // live 模式：纯数字（摄像头 index）或 /dev/video* 设备路径
    std::string video_str(VIDEO);
    bool live = (!video_str.empty() &&
                 video_str.find_first_not_of("0123456789") == std::string::npos) ||
                (video_str.rfind("/dev/video", 0) == 0);

    cv::VideoCapture cap;
    if (live && video_str.find_first_not_of("0123456789") == std::string::npos)
        cap.open(std::stoi(video_str), cv::CAP_V4L2);  // 纯数字 index
    else if (live)
        cap.open(VIDEO, cv::CAP_V4L2);                 // /dev/videoN
    else
        cap.open(VIDEO);                               // 文件路径
    if (!cap.isOpened()) { std::cerr << "Cannot open: " << VIDEO << "\n"; return 1; }

    // 从第一帧获取分辨率
    cv::Mat first_frame;
    if (!cap.read(first_frame)) { std::cerr << "Cannot read first frame\n"; return 1; }
    int h = first_frame.rows, w = first_frame.cols;
    int sw = int(w * TRACK_SCALE), sh = int(h * TRACK_SCALE);

    // file 模式：预加载全部帧
    std::vector<cv::Mat> all_frames;
    int total = 0;
    if (!live) {
        all_frames.push_back(first_frame.clone());
        cv::Mat f;
        while (cap.read(f)) all_frames.push_back(f.clone());
        cap.release();
        total = (int)all_frames.size();
    }

    // 共享内存
    SHM_H = h; SHM_W = w; SHM_SIZE = SHM_H * SHM_W * SHM_C;
    if (!setup_shm()) { std::cerr << "SHM failed\n"; return 1; }
    std::cout << "SHM ready: " << SHM_NAME << " " << SHM_W << "x" << SHM_H << std::endl;

    // 启动 Python GDino 子进程
    int pipe_to[2], pipe_from[2];
    pipe(pipe_to);
    pipe(pipe_from);

    pid_t pid = fork();
    if (pid == 0) {
        close(pipe_to[1]);
        close(pipe_from[0]);
        dup2(pipe_to[0], STDIN_FILENO);
        dup2(pipe_from[1], STDOUT_FILENO);
        close(pipe_to[0]);
        close(pipe_from[1]);
        std::string shm_arg = std::string(SHM_NAME).substr(1);
        std::string h_str = std::to_string(h);
        std::string w_str = std::to_string(w);
        execlp(PYTHON, PYTHON, "-u",
               "gdino_pipe_server.py", shm_arg.c_str(), h_str.c_str(), w_str.c_str(), nullptr);
        perror("exec failed");
        return 1;
    }

    close(pipe_to[0]);
    close(pipe_from[1]);
    to_gdino = fdopen(pipe_to[1], "w");
    from_gdino = fdopen(pipe_from[0], "r");

    std::cout << "Waiting for GDino..." << std::endl;
    char buf[256];
    while (fgets(buf, sizeof(buf), from_gdino)) {
        std::string line(buf);
        std::cerr << "  [GDino] " << line;
        if (line.find("READY") != std::string::npos) break;
    }
    std::cout << "GDino ready" << std::endl;

    // 追踪状态
    std::map<int, cv::Mat> frame_buffer;
    cv::Ptr<cv::TrackerKCF> tracker;
    bool tracking = false;
    cv::Rect rect;
    int miss_count = 0, corrections = 0, confirms = 0, det_count = 0, skipped = 0;

    // ── FILE 模式 ──────────────────────────────────────────────────────────
    if (!live) {
        std::cout << total << " frames " << w << "x" << h
                  << ", track on " << sw << "x" << sh << std::endl;

        std::map<int, std::pair<cv::Rect, std::string>> bbox_results;

        send_detect(all_frames[0], 0, h, w);
        frame_buffer[0] = all_frames[0];
        bbox_results[0] = {cv::Rect(), "WAIT"};

        auto stream_start = std::chrono::steady_clock::now();
        std::cout << "\n=== Phase 1 ===" << std::endl;

        for (int i = 1; i < total; i++) {
            auto deadline = stream_start + std::chrono::microseconds(int(i * FRAME_INTERVAL * 1e6));
            cv::Mat& frame = all_frames[i];
            auto sim_start = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::milliseconds(3));

            frame_buffer[i] = frame;
            if (frame_buffer.size() > 30) frame_buffer.erase(frame_buffer.begin());

            // deadline 超时快速路径
            auto now = std::chrono::steady_clock::now();
            if (now > deadline + std::chrono::milliseconds(1)) {
                if (has_prev && tracking) {
                    cv::Rect predicted = predict_next();
                    bbox_results[i] = {predicted, "PRED"};
                    update_momentum(predicted);
                    rect = predicted;
                } else {
                    bbox_results[i] = {cv::Rect(), "SKIP"};
                }
                skipped++;
                if (!gdino_pending) send_detect(frame, i, h, w);
                continue;
            }

            std::string tag = process_frame(frame, i, h, w, sw, sh,
                frame_buffer, tracker, tracking, rect,
                miss_count, corrections, confirms, det_count);

            // deadline 校正后补偿时钟
            if (tag == "CORR") {
                auto cost = std::chrono::steady_clock::now() - sim_start;
                if (cost > std::chrono::microseconds(int(FRAME_INTERVAL*1e6)))
                    stream_start += cost - std::chrono::microseconds(int(FRAME_INTERVAL*1e6));
            }

            bbox_results[i] = {(tracking && rect.width > 0) ? rect : cv::Rect(), tag};

            auto t_ms = std::chrono::duration<float, std::milli>(
                std::chrono::steady_clock::now() - sim_start).count();
            if (i % 30 == 0)
                printf("F%d %s corr=%d det=%d skip=%d t=%.0fms\n",
                       i, tag.c_str(), corrections, det_count, skipped, t_ms);

            auto wait = deadline - std::chrono::steady_clock::now();
            if (wait.count() > 0)
                std::this_thread::sleep_for(wait);
        }

        fprintf(to_gdino, "STOP\n"); fflush(to_gdino);
        fclose(to_gdino); fclose(from_gdino);
        waitpid(pid, nullptr, 0);

        printf("\nPhase 1: reinit=%d confirm=%d det=%d skip=%d\n",
               corrections-confirms, confirms, det_count, skipped);

        // Phase 2: Draw & Write
        std::cout << "\n=== Phase 2 ===" << std::endl;
        cv::VideoWriter writer(OUT, cv::VideoWriter::fourcc('m','p','4','v'), TARGET_FPS, cv::Size(w, h));
        for (int i = 0; i < total; i++) {
            cv::Mat vis = all_frames[i].clone();
            if (bbox_results.count(i)) {
                auto& [bbox, tag] = bbox_results[i];
                if (bbox.width > 0)
                    cv::rectangle(vis, bbox, cv::Scalar(0,255,255), 2);
                cv::putText(vis, "F" + std::to_string(i) + " " + tag,
                           cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                           cv::Scalar(255,255,255), 2);
            }
            writer.write(vis);
        }
        writer.release();
        cleanup_shm();
        std::cout << "saved → " << OUT << std::endl;

    // ── LIVE 摄像头模式 ────────────────────────────────────────────────────
    } else {
        std::cout << "Live camera " << w << "x" << h
                  << ", track on " << sw << "x" << sh << "\n"
                  << "Press 'q' or ESC to stop, recording → " << OUT << std::endl;

        cv::VideoWriter writer(OUT, cv::VideoWriter::fourcc('m','p','4','v'), TARGET_FPS, cv::Size(w, h));
        std::string win = std::string("Tracker: ") + QUERY;
        cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

        int i = 0;
        cv::Mat frame = first_frame.clone();

        while (true) {
            frame_buffer[i] = frame.clone();
            if (frame_buffer.size() > 30) frame_buffer.erase(frame_buffer.begin());

            std::string tag;
            if (i == 0) {
                send_detect(frame, 0, h, w);
                tag = "WAIT";
            } else {
                tag = process_frame(frame, i, h, w, sw, sh,
                    frame_buffer, tracker, tracking, rect,
                    miss_count, corrections, confirms, det_count);
            }

            // 绘制叠加层
            cv::Mat vis = frame.clone();
            if (tracking && rect.width > 0)
                cv::rectangle(vis, rect, cv::Scalar(0,255,255), 2);
            cv::putText(vis, "F" + std::to_string(i) + " " + tag,
                       cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                       cv::Scalar(255,255,255), 2);
            cv::putText(vis, QUERY,
                       cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                       cv::Scalar(0,255,255), 2);

            cv::imshow(win, vis);
            writer.write(vis);

            if (i % 30 == 0)
                printf("F%d %s corr=%d det=%d\n", i, tag.c_str(), corrections, det_count);

            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) break;

            i++;
            if (!cap.read(frame)) break;
        }

        fprintf(to_gdino, "STOP\n"); fflush(to_gdino);
        fclose(to_gdino); fclose(from_gdino);
        waitpid(pid, nullptr, 0);
        writer.release();
        cv::destroyAllWindows();
        cleanup_shm();
        printf("\nLive: reinit=%d confirm=%d det=%d frames=%d\n",
               corrections-confirms, confirms, det_count, i);
        std::cout << "saved → " << OUT << std::endl;
    }

    return 0;
}
