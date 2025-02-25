
<div align="center">

# MaaXuexi

基于 **[MaaFramework](https://github.com/MaaXYZ/MaaFramework)** 实现自动化完成学习强国积分任务的工具。

</div>

<p align="center">
  <img alt="license" src="https://img.shields.io/github/license/ravizhan/MaaXuexi">
  <img alt="Python" src="https://img.shields.io/badge/Python 3.12-3776AB?logo=python&logoColor=white">
  <img alt="Auther" src="https://img.shields.io/badge/code%20by-ravi-127fca">
</p>

<div align="center">

[<img src="https://api.gitsponsors.com/api/badge/img?id=892976182" height="30">](https://api.gitsponsors.com/api/badge/link?p=BmTVRa4Q8TtMZS4DMpTG3041SLDUk4W2uhU81GU9B62IsJgT2SNR1EUla6Y/Y4pUipjwNBlY2madJRzcueOYZiai/Ey00xo1lT5jz3Cp5o/bVdYejQ7BC1AnMOAoH8L+2abXXdw5dRDwMerZgdMGtQ==)
</div>

## 警告

本项目仅限用于学习研究，禁止倒卖本项目

因使用本项目造成的**任何后果**(*包括但不限于封号*)与本人无关

**本项目仅供学习交流使用，禁止用于商业用途，否则后果自负。**

**本项目仅供学习交流使用，禁止用于商业用途，否则后果自负。**

**本项目仅供学习交流使用，禁止用于商业用途，否则后果自负。**

## 功能
### 阅读文章
*稳定12分*

可自动识别文章位置，自动阅读文章
### 观看视频
*大概率7分 小概率12分*

由于前一天看过的视频，第二天不会计算分数，所以大概率只有7分 
### 每日答题
选择题一般没有问题，除非题目一页放不下，会出现识别错误，暂时无法解决

填空题一般也没有问题，除非答案有换行，会出现识别错误，暂时也无法解决
### 趣味答题
*开发中*
### webui界面
即将更新️ 🛠️
### 关于人机验证码
使用模拟器时，每日答题结束会稳定出现人机验证码，由于未知的法律风险，本项目不会提供自动识别验证码的功能。

当识别到验证码时，程序会自动停止，弹窗提醒，等待用户完成验证码后继续运行。
## 环境要求

- Python 3.10 +

- 安卓模拟器或真机

  安卓模拟器推荐使用 [雷电模拟器](https://www.ldmnq.com/) 或 [MuMu 模拟器 12](https://mumu.163.com/)
  
  请将模拟器分辨率设置为 1280x720 DPI 240

  真机暂未进行测试，请**谨慎使用**

## 使用

### AI配置

本项目使用字节跳动的 Doubao-1.5-vision-pro-32k 来解决选择题和部分填空题。
> 无广，问就是便宜而且效果不错

先到 [火山引擎](https://www.volcengine.com/) 注册账号并实名

然后到 [火山方舟控制台](https://console.volcengine.com/ark/region:ark+cn-beijing/openManagement) 开通 Doubao-1.5-vision-pro-32k 

![image-20250123170639338](https://img.ravi.top/img/4e1072e68a1f0a9892e8fb248619be4c.png)

最后到 [API Key 管理](https://console.volcengine.com/ark/region:ark+cn-beijing/apiKey) ，创建API Key并复制

![image-20250123171536042](https://img.ravi.top/img/543d40dfb2c2e28423652befabc4a3ba.png)

**关于费用**

调用一次AI大概消耗1600tokens，其中输出tokens极少，可忽略不计

按照每日答题5道题，趣味答题5道题计算，一个月大约消耗47万tokens

![](https://img.ravi.top/img/ced39313100f7e59538ea989b8d3374b.png)

一个月费用在1.4元左右

此外，火山引擎注册即送50万tokens免费推理额度，足够免费使用1个月

### 项目配置

请到 [releases](https://github.com/ravizhan/MaaXuexi/releases) 页面下载对应系统的压缩包，并解压。

用记事本打开 `config` 文件夹中的 `config.json` ，粘贴API Key

``` json
{
  "api_key": ""
}
```

### 运行
请先运行安卓模拟器，手动打开并登录学习强国APP

**如果是第一次运行，请先手动点开各个板块，确保APP不会弹出新手引导**
#### windows
下载安装 [vc_redist](https://aka.ms/vs/17/release/vc_redist.x64.exe)

双击 `MaaXuexi.exe` 即可运行
#### linux/macos
> **注意**: linux/macos均未进行测试，如有BUG请即时通过 [Issue](https://github.com/ravizhan/MaaXuexi/issues) 反馈

使用终端运行 `MaaXuexi.bin` 即可

## 许可

基于本项目使用的 [MaaFramework](https://pypi.org/project/MaaFw/) ，本项目采用 [AGPLv3协议](https://github.com/ravizhan/MaaXuexi/blob/main/LICENSE) 开源

**本项目完全免费，严禁倒卖或用于盈利目的**