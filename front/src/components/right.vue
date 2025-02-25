<template>
  <div class="col-name">实时日志</div>
  <div>
    <n-card hoverable>
      <n-button id="btn" block tertiary type="info" :data-clipboard-text="log">
        复制
      </n-button>
      <n-log class="log" :log="log" trim :rows="30" />
    </n-card>
  </div>
</template>
<script setup>
import Clipboard from 'clipboard';
import { useMessage } from "naive-ui";
import { createSocket } from '@/assets/ws.js'
import { ref, onMounted } from "vue";

const message = useMessage()
const log = ref('')
const btnCopy = new Clipboard('#btn')
btnCopy.on('success', () => {
  message.success('复制成功')
})
onMounted(() => {
  createSocket('ws://'+window.location.host+'/api/ws')
  const getsocketData = e => {
    const data = e && e.detail.data
    log.value = log.value + "\n" + data
  }
  window.addEventListener('onmessageWS', getsocketData)
})
</script>