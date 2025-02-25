// Reference: https://juejin.cn/post/7175043923709001765
let Socket = ''

export const createSocket = url => {
  Socket && Socket.close()
  if (!Socket) {
    console.log('建立websocket连接')
    Socket = new WebSocket(url)
    Socket.onmessage = onmessageWS
    Socket.onerror = onerrorWS
    Socket.onclose = oncloseWS
  } else {
    console.log('websocket已连接')
  }
}

const onerrorWS = () => {
  Socket.close()
  console.log('连接失败重连中')
  if (Socket.readyState !== 3) {
    Socket = null
    createSocket()
  }
}

const onmessageWS = e => {
  window.dispatchEvent(new CustomEvent('onmessageWS', {
    detail: {
      data: e.data
    }
  }))
}

const connecting = message => {
  setTimeout(() => {
    if (Socket.readyState === 0) {
      connecting(message)
    } else {
      Socket.send(JSON.stringify(message))
    }
  }, 1000)
}

export const sendWSPush = message => {
  if (Socket.readyState === 3) {
    Socket.close()
    createSocket()
  } else if (Socket.readyState === 1) {
    Socket.send(JSON.stringify(message))
  } else if (Socket.readyState === 0) {
    connecting(message)
  }
}

const oncloseWS = () => {
  console.log('websocket已断开....正在尝试重连')
  if (Socket.readyState !== 2) {
    Socket = null
    createSocket()
  }
}