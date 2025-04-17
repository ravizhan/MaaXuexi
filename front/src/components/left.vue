<template>
  <div class="col-name">设置</div>
  <div style="padding: 0 2.5% 0 2.5%">
    <n-card hoverable>
      <n-form
          ref="formRef"
          :model="model"
          :rules="rules"
          label-placement="left"
          label-width="auto"
          require-mark-placement="right-hanging"
      >
        <n-form-item label="API KEY" path="api_key">
          <n-input v-model:value="model.api_key" type="text" />
        </n-form-item>
        <div class="form-btn">
          <n-button strong secondary type="info" size="large" @click="Validate">
            保存
          </n-button>
        </div>
      </n-form>
    </n-card>
    <n-card style="margin-top: 1rem;" hoverable>
      <n-form-item
          label-placement="left"
          label-width="auto"
          label="设备"
          path="device">
        <n-select
            v-model:value="model.device"
            placeholder="请选择一个设备"
            :options="devices_list"
            :loading="loading"
            remote
            @click="getDevices"
        />
      </n-form-item>
      <n-button strong secondary type="info" block @click="connectDevices">
        连接
      </n-button>
    </n-card>
  </div>
</template>
<script setup>
import { useMessage } from "naive-ui";
import { ref, onMounted } from "vue";

const formRef = ref(null)
const message = useMessage()
const model = ref({
  device: null,
  api_key: '',
  endpoint: '',
})
const devices_list = ref([])
const loading = ref(false)
const rules = {
  api_key: {
    required: true,
    trigger: ["blur", "input"],
    validator(rule, value) {
      if (!value) {
        return new Error("请输入 API KEY");
      } else if (!/^sk-[a-z]{48}$/.test(value)) {
        return new Error("格式错误");
      }
      return true;
    }
  },
  endpoint: {
    required: true,
    trigger: ["blur", "input"],
    validator(rule, value) {
      if (!value) {
        return new Error("请输入 endpoint");
      } else if (!/^ep-\d{14}-\w{5}$/.test(value)) {
        return new Error("格式错误");
      }
      return true;
    }
  }
}

function Validate(e) {
  e.preventDefault();
  formRef.value?.validate((errors) => {
    if (!errors) {
      fetch('/api/settings', {
        method: 'POST',
        body: JSON.stringify(model.value),
        headers: {
          'Content-Type': 'application/json'
        }
      })
      message.success("保存成功");
    }
  });
}

function getDevices() {
  devices_list.value = []
  loading.value = true
  fetch('/api/get_device', {
    method: 'GET'
  }).then(res => res.json()).then(data => {
    if (data["status"] === "failed") {
      message.error(data["message"])
      loading.value = true
      return
    }
    const devices_data = data["devices"]
    for (let i = 0; i < devices_data.length; i++) {
      console.log(devices_data)
      devices_list.value.push({ label: devices_data[i]["name"]+" "+devices_data[i]["address"], value: devices_data[i] })
    }
    loading.value = false
  })
}

function connectDevices() {
  if (model.value.device === null) {
    message.error('请选择一个设备')
  } else {
    fetch('/api/connect_device', {
      method: 'POST',
      body: JSON.stringify(model.value.device),
      headers: {
        'Content-Type': 'application/json'
      }
    }).then(res => res.json()).then(data => {
      if (data["status"] === "success") {
        message.success('连接成功')
      } else {
        message.error('连接失败')
      }
    })
  }
}

onMounted(() => {
  fetch('/api/settings', {
    method: 'GET'
  }).then(res => res.json()).then(data => {
    model.value.api_key = data["api_key"]
    model.value.endpoint = data["endpoint"]
  })
})
</script>