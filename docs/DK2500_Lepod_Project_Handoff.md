# DK-2500 + Lepod Pro (ER3) 项目交接文档

## 1. 项目背景与任务

### 已完成的任务

**Todo1: 购买无线网卡，并在 Ubuntu 24.04 环境下测试蓝牙连接功能 ✅**

- 最终使用 Intel AX201NGW（CNVi 版本）安装在 M.2 E-Key 槽位
- Wi-Fi 和蓝牙均已测试通过
- 已成功通过蓝牙连接 Lepod 0020，读取电量、接收心率 notify 数据

**Todo2: 移植 ER3 的 SDK，进一步测试数据读取和基础采集 ✅ 已完成**

- SDK 为 Android AAR 格式，无法直接用于 Ubuntu
- 已用 Python（bleak 库）逆向实现了完整的 BLE 通信协议
- 已成功实现 8 通道 250Hz ECG 实时波形采集、解压缩、CSV 导出和可视化
- 详见下方第 5 节

---

## 2. 硬件环境

### DK-2500 开发套件

- **处理器**: Intel Core Ultra 5 225U (Arrow Lake)
- **内存**: 8GB DDR5
- **SSD**: 128GB
- **系统**: Ubuntu 24.04 Desktop，内核 6.17.0-23-generic
- **网卡**: Intel AX201NGW（M.2 E-Key，CNVi 模式）
- **以太网**: 4 × Intel i210 GbE LAN
- **SSH 访问**: 通过网线静态 IP 连接笔记本电脑

### 无线网卡说明

- E-Key 槽位标记为 M2_WIFI-CNVi，仅支持 CNVi 和 USB 信号，**不提供标准 PCIe 通道**
- 因此 AX210NGW（独立 PCIe 网卡）的 Wi-Fi 功能无法使用（蓝牙走 USB 可用）
- 已更换为 AX201NGW（CNVi 网卡），Wi-Fi 和蓝牙均正常

### Wi-Fi 连接信息

- **SSID**: BISTU-802.1X
- **认证方式**: WPA-EAP, PEAP + MSCHAPv2
- **账号**: 2024011182
- **密码**: 47p7ufhg+
- **连接命令**:

```bash
sudo nmcli connection add type wifi con-name "BISTU-802.1X" ifname wlo1 ssid "BISTU-802.1X" \
  wifi-sec.key-mgmt wpa-eap \
  802-1x.eap peap \
  802-1x.phase2-auth mschapv2 \
  802-1x.identity "2024011182" \
  802-1x.password "47p7ufhg+"
sudo nmcli connection up "BISTU-802.1X"
```

---

## 3. Lepod Pro (ER3) 设备信息

### 基本信息

- **设备名称**: Lepod 0020
- **BLE MAC 地址**: D3:EA:37:4F:E2:38（random 类型）
- **设备序列号**: 40110002
- **电量**: 65%（最近测试时）
- **连接方式**: BLE，无需配对，直接 connect 即可

### BLE 服务列表

| 服务 | UUID | 说明 |
|------|------|------|
| Generic Access Profile | 00001800-0000-1000-8000-00805f9b34fb | 标准 GAP |
| Generic Attribute Profile | 00001801-0000-1000-8000-00805f9b34fb | 标准 GATT |
| Heart Rate | 0000180d-0000-1000-8000-00805f9b34fb | 标准心率服务 |
| Battery Service | 0000180f-0000-1000-8000-00805f9b34fb | 电量服务 |
| Device Information | 0000180a-0000-1000-8000-00805f9b34fb | 设备信息 |
| Nordic Semiconductor | 0000fe59-0000-1000-8000-00805f9b34fb | DFU 固件升级 |
| **Vendor Specific (乐普私有)** | **14839ac4-7d7e-415c-9a42-167340cf2339** | **ECG 数据通道** |

### 乐普私有服务（关键）

- **Service UUID**: 14839AC4-7D7E-415C-9A42-167340CF2339
- **Write Characteristic**: 8B00ACE7-EB0B-49B0-BBE9-9AEE0A26E1A3（向设备发送命令）
- **Notify Characteristic**: 0734594A-A8E7-4B1A-A6B1-CD5243059A57（接收设备数据）

### GATT 路径（bluetoothctl 中验证过的）

```
# 电量
/org/bluez/hci0/dev_D3_EA_37_4F_E2_38/service0014/char0015

# 心率 (notify)
/org/bluez/hci0/dev_D3_EA_37_4F_E2_38/service000e/char000f

# 乐普私有服务
/org/bluez/hci0/dev_D3_EA_37_4F_E2_38/service0023/char0024  (Write)
/org/bluez/hci0/dev_D3_EA_37_4F_E2_38/service0023/char0026  (Notify)
```

---

## 4. Lepod BLE 协议（逆向实现）

> **重要**: 以下协议细节是通过逆向 Android SDK 源码 + 实机探测验证得出的，并非来自官方文档。
> 原始 SDK 中的 ER3 命令（0x05 RT_DATA、0x06 RT_DATA）对 Lepod 无效，Lepod 使用不同的命令集。

### 通用包格式

所有命令和响应均使用相同的帧结构：

```
Byte 0:    0xA5              (同步头)
Byte 1:    CMD_ID            (命令 ID)
Byte 2:    ~CMD_ID           (CMD_ID 按位取反，用于校验)
Byte 3:    0x00              (保留 / 包类型)
Byte 4:    SEQ_NO            (序列号，0-254 循环)
Byte 5:    LEN_LO            (payload 长度低字节，小端)
Byte 6:    LEN_HI            (payload 长度高字节)
Byte 7..N: PAYLOAD           (0 或多个数据字节)
Last byte: CRC8              (对前面所有字节计算 CRC-8)
```

总包长 = 8 + payload_length。CRC-8 多项式 0x07，初始值 0x00。

### Lepod 命令列表（已验证）

| CMD ID | 功能 | Payload | 响应 |
|--------|------|---------|------|
| `0xE1` | 获取设备信息 | 无 | 60B 设备信息 |
| `0x00` | 获取配置 | 无 | 1B 配置模式 |
| `0x04` | 设置配置 | 1B: 模式(0/1/2) | 空 ACK |
| `0x02` | 获取实时参数 | 无 | 40B 参数（无波形） |
| `0x03` | 获取实时数据 | 无 | 40B 参数 + 压缩波形 |
| **`0x0A`** | **开始录制** | **无** | **空 ACK** |
| **`0x07`** | **停止录制** | **无** | **空 ACK** |
| `0x08` | 未知（返回空 ACK） | 无 | 空 ACK |
| `0x09` | 获取状态 | 无 | 20B 状态 |
| `0xE2` | 重置 | 无 | 空 ACK |

> **注意**: SDK 文档中的 `0x05`（ER3_RT_DATA）和 `0x06`（RT_DATA）对 Lepod **无响应**。
> Lepod 的实时数据命令是 `0x03`。

### 实时数据响应格式（CMD 0x03）

响应 payload 结构：

**参数段（40 字节，offset 0-39）：**

| Offset | 长度 | 字段 | 说明 |
|--------|------|------|------|
| 0 | 1 | status | 状态标志 |
| 1-2 | 2 | year | 年份 uint16_LE（如 0x07EA=2026） |
| 3 | 1 | month | 月 |
| 4 | 1 | day | 日 |
| 5 | 1 | hour | 时 |
| 6 | 1 | minute | 分 |
| 7 | 1 | second | 秒 |
| 8-9 | 2 | hr | 心率 uint16_LE（设备未佩戴时为 0） |
| 10-12 | 3 | 其他生理参数 | — |
| 13 | 1 | battery | 电量百分比（0-100） |
| 14-39 | 26 | 其他参数/保留 | 含 recordTime, lead 类型等 |

**波形头（10 字节，offset 40-49）：**

| Offset | 长度 | 字段 | 说明 |
|--------|------|------|------|
| 40-43 | 4 | reserved | 保留（全 0） |
| 44-47 | 4 | firstIndex | 采样起始索引 uint32_LE |
| 48-49 | 2 | sampleCount | 本次采样点数 uint16_LE（通常 250-325） |

**波形数据（offset 50+，8 通道差分压缩）：**

解压缩算法：

```
[0-1]:   头部 0x80 0xFF
[2-17]:  8 × int16_LE 各通道初始值
[18+]:   交错 8-bit delta 编码：
         - 正常字节: int8 delta，加到对应通道前一个值
         - 0x80 字节: escape 标记，下一字节才是实际 int8 delta
```

8 个通道对应：V6, I, II, V1, V2, V3, V4, V5

电压转换：`mV = raw_value × 0.002467`

### 采集流程

```
1. 扫描并连接 Lepod 设备
2. 订阅 Notify Characteristic
3. 发送 0x0A（开始录制）
4. 等待 2-3 秒让缓冲区积累数据
5. 每秒发送 0x03（读取实时数据），每次获得 ~250 采样点 × 8 通道
6. 解压缩波形数据，累积到各通道数组
7. 完成后发送 0x07（停止录制）
```

### CRC-8 查表

多项式 0x07，初始值 0x00，完整查找表见 `workspace/lepod_ecg.py`。

---

## 5. Python 采集工具

### 环境配置

```bash
# 虚拟环境（已创建）
source ~/workspace/lepod_env/bin/activate

# 依赖
pip install bleak matplotlib numpy
```

### 采集脚本

```bash
# 采集 30 秒 ECG（默认）
python3 ~/workspace/lepod_ecg.py

# 采集 60 秒
python3 ~/workspace/lepod_ecg.py 60
```

**输出文件**（在 `~/workspace/` 下）：
- `ecg_YYYYMMDD_HHMMSS.csv` — 8 通道 ECG 数据（采样索引、时间、各通道 mV 值）
- `ecg_YYYYMMDD_HHMMSS_8ch.png` — 8 通道波形图
- `ecg_YYYYMMDD_HHMMSS_leadII.png` — Lead II 单通道波形图

### 主要源码文件

| 文件 | 用途 |
|------|------|
| `workspace/lepod_ecg.py` | **主采集脚本**（最终版，含完整协议实现） |
| `workspace/lepod_decode.py` | 离线解码工具（从 capture JSON 解码波形） |
| `workspace/lepod_probe.py` | 协议探测工具（扫描所有命令 ID） |
| `workspace/lepod_debug.py` | 调试工具（原始通知数据 dump） |
| `workspace/lepod_stream_test.py` | 持续采集测试（验证 0x0A 启动录制） |
| `workspace/capture_*.json` | 原始帧数据存档 |

### 已验证的采集结果

- **采样率**: 250 Hz，实测稳定
- **通道数**: 8（V6, I, II, V1, V2, V3, V4, V5）
- **电压范围**: ±3 mV（正常 ECG 范围）
- **持续时间**: 已验证 30 秒持续采集，每秒约 250 个新采样点
- **QRS 波形**: 在 Lead II 等通道清晰可见心跳尖峰

---

## 6. 已验证的标准服务测试结果

### 标准心率服务

```
# 心率 notify 数据示例（未佩戴时）
04 00
# 04 = 格式标志（未检测到皮肤接触）
# 00 = 心率值 0
```

### 电量读取

```
# 电量值
41 (hex) = 65 (decimal) = 65%
```

---

## 7. 已知问题与后续方向

### 已知问题

1. **帧边界噪声**: 相邻帧之间偶有小幅跳变，因为每帧有独立的初始值。可通过高通滤波或帧间平滑处理
2. **信号质量**: 波形含基线漂移和运动伪影，与 Lepod 佩戴位置和皮肤接触质量有关
3. **通道映射**: 8 通道命名（V6/I/II/V1-V5）基于 SDK lead_type=0x00（12 导联模式），Lepod 胸贴的实际物理导联映射可能不同

### 后续可做的工作

1. **信号处理**: 添加带通滤波（0.5-40Hz）、基线漂移去除、50Hz 工频干扰滤波
2. **R 峰检测**: 实现 Pan-Tompkins 等算法，自动计算心率
3. **实时显示**: 基于 matplotlib 动画或 PyQt 实现实时滚动波形显示
4. **长时记录**: 优化存储格式（如 EDF/HDF5），支持数小时连续记录
5. **多设备支持**: 扩展到其他 Viatom/Lepu 系列设备（ER1、ER2 等）
