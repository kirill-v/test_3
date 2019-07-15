#include "thread_pool.h"

ThreadPool::ThreadPool(const unsigned int threads)
    : kQueueMaxSize_(threads), join_requested_(false) {
  threads_.reserve(threads);
  for (auto i = 0U; i < threads; ++i) {
    threads_.emplace_back(std::bind(&ThreadPool::worker, this));
  }
}

ThreadPool::~ThreadPool() { Join(); }

void ThreadPool::Join() {
  LockType lock(mutex_);
  join_requested_ = true;
  lock.unlock();

  for (auto& t : threads_) {
    new_task_.notify_all();
    if (t.joinable()) {
      t.join();
    }
  }
}

void ThreadPool::RunTask(TaskType&& task) {
  LockType lock(mutex_);

  if (tasks_.size() >= kQueueMaxSize_) {
    free_worker_.wait(lock,
                      [this]() { return tasks_.size() < kQueueMaxSize_; });
  }

  tasks_.push(std::move(task));
  lock.unlock();

  new_task_.notify_one();
}

void ThreadPool::worker() {
  while (true) {
    LockType lock(mutex_);

    if (tasks_.empty()) {
      if (join_requested_) {
        return;
      }
      new_task_.wait(lock,
                     [this] { return !tasks_.empty() || join_requested_; });
      if (tasks_.empty()) {
        return;
      }
    }

    TaskType task = tasks_.front();
    tasks_.pop();
    lock.unlock();
    task();

    free_worker_.notify_one();
  }
}
