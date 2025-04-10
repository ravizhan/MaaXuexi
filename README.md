
<div align="center">

# MaaXuexi

基于 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 实现自动化完成学习强国积分任务的工具。

</div>

<p align="center">
  <img alt="license" src="https://img.shields.io/github/license/ravizhan/MaaXuexi">
  <img alt="Python" src="https://img.shields.io/badge/Python 3.12-3776AB?logo=python&logoColor=white">
  <img alt="Auther" src="https://img.shields.io/badge/code%20by-ravi-127fca">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ravizhan/MaaXuexi">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/ravizhan/MaaXuexi">
  <img alt="GitHub Repo Downloads" src="https://img.shields.io/github/downloads/ravizhan/MaaXuexi/total?style=social">
</p>

## 警告

**本项目仅供学习交流使用**

**本项目仅供学习交流使用**

**本项目仅供学习交流使用**

因使用本项目造成的**任何后果**(*包括但不限于**封号***)与本人无关

因使用本项目造成的**任何后果**(*包括但不限于**封号***)与本人无关

因使用本项目造成的**任何后果**(*包括但不限于**封号***)与本人无关

**本项目仅供学习交流使用，禁止倒卖或用于商业用途，否则后果自负。**

## 功能
### 阅读文章
*稳定12分*

可自动识别文章位置，自动阅读文章
### 观看视频
*大概率7分 小概率12分*

由于前一天看过的视频，第二天不会计算分数，所以大概率只有7分 
### 每日答题
选择题和填空题均可自动识别

选择题基本全对

填空题一般也没有问题，除非答案有换行，会出现识别错误，暂时也无法解决
### 趣味答题
*开发中 🛠️*
### WebUI界面
基于Naive UI开发的WebUI界面，简洁美观
### 关于人机验证码
使用模拟器时，每日答题结束会稳定出现人机验证码，由于未知的法律风险，本项目不会提供自动识别验证码的功能。

当识别到验证码时，程序会自动停止，弹窗提醒，等待用户完成验证码后继续运行。
## 环境要求

- Python 3.10 +

- 安卓模拟器

  安卓模拟器推荐使用 [雷电模拟器](https://www.ldmnq.com/) 或 [MuMu 模拟器 12](https://mumu.163.com/)
  
  请将模拟器分辨率设置为 1280x720 DPI 240 横屏模式

## 使用

### 下载
请前往 [Release](https://github.com/ravizhan/MaaXuexi/releases) 下载对应系统的压缩包，并解压
> **注意**: 
> 
> windows和linux暂时均只提供x86_64架构版本
> 
> macos只提供arm64架构版本

### AI配置

本项目使用[硅基流动](https://siliconflow.cn/)平台提供的 Qwen2.5-VL-7B-Instruct 来解决选择题和部分填空题。
> 使用[邀请链接注册](https://cloud.siliconflow.cn/i/krMRM7qS)可获赠14元余额，可长期使用

注册登录后，在 [API密钥](https://cloud.siliconflow.cn/account/ak) 页面，点击 `新建API密钥` 按钮，新建密钥然后复制

**关于费用**

调用一次AI大概消耗800tokens，其中输出tokens极少，可忽略不计

按照每日答题5道题，趣味答题5道题计算，一个月大约消耗24万tokens

目前 Qwen2.5-VL-7B-Instruct 的价格为 0.35元/百万tokens，一个月仅需0.085元。
### 运行
#### windows
下载安装 [vc_redist](https://aka.ms/vs/17/release/vc_redist.x64.exe)

双击 `MaaXuexi.exe` 即可运行
#### linux/macos
> **注意**: linux/macos均未进行测试，如有BUG请即时通过 [Issue](https://github.com/ravizhan/MaaXuexi/issues) 反馈

使用终端运行 `MaaXuexi.bin` 即可
### WebUI设置
启动成功后会自动打开浏览器，跳转到 `http://127.0.0.1:8000/`

如果没有自动打开，请手动输入地址。

首次打开时左上角会弹窗请求通知权限，请点击 `允许` 以确保正常发送通知

![](https://img.ravi.top/img/470c7498ff549abdb61f820522ace6f9.png)

1. 在 `设置` 中填入刚刚复制的API Key，点击 `保存`
2. 在 `设备栏` 下拉选择模拟器，点击 `连接`
3. 在 `任务栏` 勾选需要完成的任务，点击 `开始任务`

连接设备后会自动打开APP，请确保APP已登录再开始任务
> **注意**：目前mumu模拟器会出现分辨率识别错误的情况，请手动打开APP后再连接设备

**如果是第一次运行，请先手动点开各个板块，确保APP不会弹出新手引导**

> **注意**: 执行 `每日答题` 任务时，**请留意系统通知**
> 
> 当出现自动答题失败或弹出验证码时，会要求**人工接管**
> 
## 声明

基于本项目使用的 [MaaFramework](https://pypi.org/project/MaaFw/) 和 [ultralytics](https://github.com/ultralytics/ultralytics) (YOLO模型) ，本项目采用 [AGPLv3协议](https://github.com/ravizhan/MaaXuexi/blob/main/LICENSE) 开源

**本项目完全免费，严禁倒卖或用于盈利目的**

## Star History

<a href="https://www.star-history.com/#ravizhan/MaaXuexi&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ravizhan/MaaXuexi&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ravizhan/MaaXuexi&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ravizhan/MaaXuexi&type=Date" />
 </picture>
</a>