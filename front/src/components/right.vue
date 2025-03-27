<template>
  <div class="col-name">实时日志</div>
  <div>
    <n-card hoverable>
      <n-button id="btn" block tertiary type="info" :data-clipboard-text="log">
        复制
      </n-button>
      <n-log class="log" ref="logInstRef" :log="log" trim :rows="30" />
    </n-card>
  </div>
</template>
<script setup>
import Clipboard from 'clipboard';
import { useMessage, useDialog } from "naive-ui";
import { createSocket } from '@/assets/ws.js'
import { ref, onMounted, watchEffect,nextTick } from "vue";

const dialog = useDialog()
const message = useMessage()
const log = ref('')
const logInstRef = ref(null)
const btnCopy = new Clipboard('#btn')
btnCopy.on('success', () => {
  message.success('复制成功')
})

function is_now(date_string) {
  function formatDate(date) {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
  }
  return date_string === formatDate(new Date());
}

onMounted(() => {
  createSocket()
  const getsocketData = e => {
    const data = e && e.detail.data
    log.value = log.value + "\n" + data
    if (data.includes("请求接管") && is_now(data.split(" ")[0]+" "+data.split(" ")[1])) {
      message.warning(data)
      new Notification('请求接管', {
        body: data + "\n" + "完成接管后请点击确定"
      })
      dialog.warning({
        title: '请求接管',
        content: data + "\n" + "完成接管后请点击确定",
        positiveText: '确定',
        closable: false,
        maskClosable: false,
        onPositiveClick: () => {
          fetch('/api/continue', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            }
          }).then(res => res.json()).then(data => {
            if (data["status"] === "success") {
              message.success('正在恢复任务')
            } else {
              message.error(data["message"])
            }
          })
        }
      })
    }
  }
  window.addEventListener('onmessageWS', getsocketData)
  watchEffect(() => {
    if (log.value) {
      nextTick(() => {
        logInstRef.value?.scrollTo({ position: "bottom", silent: true });
      });
    }
  });
})
</script>