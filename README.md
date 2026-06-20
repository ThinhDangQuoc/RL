# Học Sâu Tăng Cường Trong Giao Dịch Chứng Khoán Tự Động: Chiến Lược Ensemble Trên Thị Trường Việt Nam (HOSE)

Dự án này áp dụng phương pháp Học sâu Tăng cường (Deep Reinforcement Learning - DRL) để xây dựng hệ thống giao dịch chứng khoán tự động trên thị trường chứng khoán Việt Nam (sàn HOSE). Hệ thống huấn luyện và kết hợp 3 thuật toán Actor-Critic tiêu biểu: **PPO** (Proximal Policy Optimization), **A2C** (Advantage Actor-Critic), và **DDPG** (Deep Deterministic Policy Gradient), kết hợp với cơ chế **Ensemble** linh hoạt và **Chỉ số Hỗn loạn** (Turbulence Index) để quản trị rủi ro khủng hoảng.

---

## 1. Giới thiệu Bài toán & Kiến trúc Hệ thống

### Mô hình hóa MDP (Markov Decision Process)
* **Không gian trạng thái (State Space - 91 chiều)**: 
  * Số dư tiền mặt hiện tại (1 chiều).
  * Số lượng cổ phiếu đang nắm giữ của 15 mã trong danh mục (15 chiều).
  * Giá đóng cửa điều chỉnh của 15 mã cổ phiếu (15 chiều).
  * 4 chỉ báo kỹ thuật phổ biến cho mỗi mã cổ phiếu: MACD, RSI, CCI, ADX ($15 \times 4 = 60$ chiều).
* **Không gian hành động (Action Space - 15 chiều)**: Vector liên tục trong đoạn $[-1, 1]^{15}$ đại diện cho hành động Bán (âm), Giữ (0), hoặc Mua (dương) đối với từng mã cổ phiếu, ràng buộc giao dịch tối đa $h_{\max} = 100$ cổ phiếu mỗi lệnh.
* **Hàm phần thưởng (Reward)**: Tối đa hóa tổng giá trị tài sản ròng (Portfolio Value) và giảm thiểu mức sụt giảm tài sản cực đại (Max Drawdown).

### Ràng buộc Thị trường Việt Nam (HOSE)
* **Chu kỳ thanh toán T+3**: Mô phỏng thực tế dòng tiền giao dịch, tiền bán chứng khoán chỉ khả dụng sau 3 ngày giao dịch.
* **Biên độ trần sàn $\pm 7\%$**: Ngăn chặn các hành vi mua giá trần hoặc bán giá sàn khi cổ phiếu chạm biên độ giao dịch hàng ngày.
* **Phí giao dịch thực tế**: Thiết lập ở mức $0.1\%$ trên mỗi giá trị giao dịch.

---

## 2. Cấu trúc Thư mục Dự án

```text
├── config/             # Cấu hình danh mục cổ phiếu, tham số huấn luyện & kiểm thử
│   └── config.py
├── data/               # Dữ liệu lịch sử 15 mã cổ phiếu HOSE (2014-2025)
│   ├── processed2.csv  # Dữ liệu huấn luyện và kiểm chứng
│   └── test.csv        # Dữ liệu kiểm thử out-of-sample
├── env/                # Môi trường giả lập giao dịch chứng khoán (Gym environments)
│   ├── EnvMultipleStock_train.py
│   ├── EnvMultipleStock_validation.py
│   └── EnvMultipleStock_trade.py
├── model/              # Logic huấn luyện tác tử và bộ chọn Ensemble
│   └── models.py
├── preprocessing/      # Tiền xử lý dữ liệu và tính toán chỉ báo kỹ thuật
│   └── preprocessors.py
├── presentation/       # Tệp nguồn Slide thuyết trình LaTeX Beamer và bản PDF hoàn chỉnh
│   ├── presentation.tex
│   └── presentation.pdf
├── figs/               # Các biểu đồ và sơ đồ phục vụ slide và báo cáo
├── results/            # Kết quả lưu trữ chạy thực nghiệm (CSV/PNG)
├── logs/               # Nhật ký ghi chép quá trình chạy mô hình
├── scripts/            # Các công cụ/script bổ trợ tính toán và vẽ biểu đồ
├── run_DRL.py          # Script chính để chạy huấn luyện và kiểm thử chiến lược
├── generate_eval_plots.py # Script tự động sinh toàn bộ biểu đồ thực nghiệm
├── requirements.txt    # Danh sách các thư viện Python bắt buộc
└── README.md           # Hướng dẫn dự án này
```

---

## 3. Danh mục Cổ phiếu & Thiết lập Thực nghiệm

### Danh mục 15 cổ phiếu Blue-chip (sàn HOSE):
* **Ngân hàng**: ACB, BID, CTG, MBB, SHB, STB, VCB
* **Công nghệ**: FPT
* **Năng lượng**: GAS
* **Thép/Sản xuất**: HPG
* **Hàng tiêu dùng**: MSN, VNM
* **Bất động sản**: VIC
* **Chứng khoán**: SSI
* **Bảo hiểm**: BVH

### Phân chia dữ liệu:
* **In-sample (Huấn luyện & Kiểm chứng)**: Từ `2014-01-02` đến `2020-12-31`.
* **Out-of-sample (Kiểm thử thực tế)**: Từ `2021-01-04` đến `2025-05-30` (Trải qua giai đoạn Up-trend mạnh 2021, Down-trend khốc liệt 2022, và phân hóa phục hồi 2023-2025).

---

## 4. Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu hệ thống:
* Hỗ trợ hệ điều hành Linux (Ubuntu khuyên dùng).
* Phiên bản Python khuyến nghị: `Python 3.6` hoặc `Python 3.7`.

### Cài đặt thư viện:
Cài đặt các gói phụ thuộc hệ thống (cho MPI và TensorFlow 1.15):
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
```
Cài đặt các thư viện Python:
```bash
pip install -r requirements.txt
```

### Chạy hệ thống:
1. **Huấn luyện và Giao dịch**:
   Chạy script chính để huấn luyện các tác tử DRL theo cơ chế cửa sổ cuốn (Growing Window) định kỳ hàng quý (63 ngày) và thực hiện giao dịch:
   ```bash
   python run_DRL.py
   ```
2. **Vẽ biểu đồ Thực nghiệm**:
   Tự động tổng hợp dữ liệu từ thư mục `results/` và kết xuất các biểu đồ trực quan lưu trong thư mục `figs/`:
   ```bash
   python generate_eval_plots.py
   ```

---

## 5. Kết quả Thực nghiệm Chính (2021--2025)

Bảng dưới đây tóm tắt hiệu năng giao dịch thực tế của các tác tử trên tập dữ liệu kiểm thử out-of-sample:

| Chỉ số Hiệu năng | Tác tử A2C | Tác tử PPO | Tác tử DDPG | Chiến lược Ensemble |
| :--- | :---: | :---: | :---: | :---: |
| **Lợi nhuận lũy kế** | 21.32% | 38.78% | **49.66%** | 35.90% |
| **Lợi nhuận năm (Annualized Return)** | 6.74% | 10.07% | **11.77%** | 9.41% |
| **Độ biến động năm (Annual Vol)** | 19.49% | 19.33% | **18.30%** | 18.57% |
| **Tỷ số Sharpe (Sharpe Ratio)** | 0.3458 | 0.5212 | **0.6432** | 0.5066 |
| **Mức sụt giảm cực đại (Max DD)** | -27.51% | -24.69% | **-16.57%** | -22.87% |

### Nhận xét quan trọng:
1. **DDPG tối ưu nhất**: Đạt Sharpe cao nhất (`0.6432`) và kiểm soát rủi ro drawdown tốt nhất (`-16.57%`) nhờ cơ chế nhiễu khám phá OU Noise phù hợp và khả năng tối ưu hóa trong không gian hành động liên tục ổn định.
2. **Ensemble cân bằng**: Đạt Sharpe ổn định (`0.5066`) và lợi nhuận tích lũy tốt. Tuy nhiên, hiệu năng thấp hơn DDPG đơn lẻ do chịu ảnh hưởng bởi **độ trễ lựa chọn** (Selection Lag) khi thị trường đảo chiều nhanh và **chi phí ma sát tái cơ cấu danh mục** ($0.1\%$) khi chuyển đổi qua lại giữa các tác tử mỗi quý.
3. **Chỉ số Hỗn loạn (Turbulence Index)**: Hoạt động hiệu quả như một hệ thống cảnh báo sớm, kích hoạt chế độ phòng vệ đưa danh mục đầu tư về tiền mặt giúp tránh khỏi các cú sập mạnh của thị trường (điển hình là đợt sụt giảm năm 2022).
