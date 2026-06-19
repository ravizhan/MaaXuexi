<template>
  <div class="col-name">任务列表</div>
  <div style="padding: 0 2.5% 0 2.5%">
    <n-card hoverable>
      <n-list>
        <n-list-item>
          <n-checkbox
              v-model:checked="tasks.read"
              size="large"
              label="选读文章"
          />
        </n-list-item>
        <n-list-item>
          <n-checkbox
              v-model:checked="tasks.watch"
              size="large"
              label="视听学习"
          />
        </n-list-item>
        <n-list-item>
          <n-checkbox
              v-model:checked="tasks.daily"
              size="large"
              label="每日答题"
          />
          <div style="display: flex; align-items: center; margin-top: 4px; margin-left: 24px; gap: 6px;">
            <n-switch v-model:value="tasks.fast_answer" size="small" />
            <n-tooltip trigger="hover">
              <template #trigger>
                <span style="font-size: 13px; color: #999; cursor: help;">极速答题 (beta)</span>
              </template>
              基于一些题目特征，快速选出答案，可能存在答错风险
            </n-tooltip>
          </div>
        </n-list-item>
        <n-list-item>
          <n-checkbox
              v-model:checked="tasks.fun"
              size="large"
              label="趣味答题"
              disabled
          />
        </n-list-item>
      </n-list>
      <n-flex class="form-btn" justify="center">
        <n-button strong secondary type="info" size="large" @click="startTask">
          开始任务
        </n-button>
        <n-button strong secondary type="info" size="large" @click="stopTask">
          中止任务
        </n-button>
      </n-flex>
    </n-card>
  </div>
</template>
<script setup>
import { ref } from "vue";
import { useMessage } from "naive-ui";

const message = useMessage()
const tasks = ref({
  read: false,
  watch: false,
  daily: false,
  fun: false,
  fast_answer: false
})

function startTask() {
  const task_list = []
  for (const key in tasks.value) {
    if (tasks.value[key]) {
      if (key === 'read') {
        task_list.push('选读文章')
      } else if (key === 'watch') {
        task_list.push('视听学习')
      } else if (key === 'daily') {
        task_list.push('每日答题')
      } else if (key === 'fun') {
        task_list.push('趣味答题')
      }
    }
  }
  fetch('/api/start', {
    method: 'POST',
    body: JSON.stringify({"tasklist":task_list, "fast_answer":tasks.value.fast_answer}),
    headers: {
      'Content-Type': 'application/json'
    }
  }).then(res => res.json()).then(data => {
    if (data["status"] === "success") {
      message.success('任务开始')
    } else {
      message.error(data["message"])
    }
  })
}

function stopTask() {
  fetch('/api/stop', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    }
  }).then(res => res.json()).then(data => {
    if (data["status"] === "success") {
      message.success('正在中止任务，请稍后')
    } else {
      message.error(data["message"])
    }
  })
}
</script>