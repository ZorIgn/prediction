import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# ========== 数据加载与预处理类 ==========
class StockPredictor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.preprocess_data()
        
    def preprocess_data(self):
        # 转换日期格式并设为索引（假设trade_date格式为YYYYMMDD）
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'], format='%Y%m%d')
        self.df.set_index('trade_date', inplace=True)
        
        # 添加技术指标
        self.df['MA5'] = self.df['close'].rolling(5).mean()    # 5日均线
        self.df['MA20'] = self.df['close'].rolling(20).mean()  # 20日均线
        self.df['RSI'] = self._calculate_rsi(14)               # 14日RSI
        self.df.dropna(inplace=True)  # 删除空值
        
    def _calculate_rsi(self, window):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

# ========== 模型训练与预测类 ==========
class StockModel:
    def __init__(self, data):
        self.data = data
        self.scaler = StandardScaler()
        
    def prepare_data(self, n_steps=5):
        # 选择特征和标签
        features = self.data[['open', 'high', 'low', 'vol', 'MA5', 'MA20', 'RSI']]
        target = self.data['close']
        
        # 标准化特征
        scaled_features = self.scaler.fit_transform(features)
        
        # 创建时间序列数据集
        X, y = [], []
        for i in range(len(scaled_features)-n_steps):
            X.append(scaled_features[i:i+n_steps])
            y.append(target.iloc[i+n_steps])
        return np.array(X), np.array(y)
    
    def train_svr(self, test_size=0.2):
        X, y = self.prepare_data()
        split = int(len(X)*(1-test_size))
        
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = SVR(kernel='rbf', C=100, gamma=0.1)
        model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
        # 评估模型
        pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        mse = mean_squared_error(y_test, pred)
        print(f"[模型评估] SVR测试集MSE: {mse:.2f}")
        return model
    
    def train_arima(self):
        model = ARIMA(self.data['close'], order=(5,1,0))  # 根据数据调整(p,d,q)
        model_fit = model.fit()
        print(model_fit.summary())
        return model_fit

# ========== 主流程：可视化与预测 ==========
def main(file_path):
    # 初始化数据处理器
    stock = StockPredictor(file_path)
    model = StockModel(stock.df)
    
    # 训练模型
    print("===== 开始训练模型 =====")
    svr_model = model.train_svr()
    arima_model = model.train_arima()
    
    # 生成未来7个交易日日期
    last_date = stock.df.index[-1]
    future_dates = pd.date_range(last_date, periods=7+1)[1:]  # 预测未来7天
    
    # SVR预测（需滚动窗口）
    n_steps = 5
    last_features = model.scaler.transform(stock.df[['open', 'high', 'low', 'vol', 'MA5', 'MA20', 'RSI']][-n_steps:])
    svr_pred = []
    current_window = last_features.copy()
    for _ in range(7):
        next_pred = svr_model.predict(current_window.reshape(1, -1))[0]
        svr_pred.append(next_pred)
        # 模拟滚动更新（实际场景需接入新数据）
        current_window = np.roll(current_window, -1, axis=0)
        current_window[-1] = [np.nan]*7  # 用实际数据替换np.nan
    
    # ARIMA预测
    arima_pred = arima_model.forecast(steps=7)
    
    # ===== 可视化结果 =====
    plt.figure(figsize=(15,6))
    plt.plot(stock.df.index, stock.df['close'], label='历史价格', color='blue')
    plt.plot(future_dates, svr_pred, label='SVR预测', marker='o', linestyle='--')
    plt.plot(future_dates, arima_pred, label='ARIMA预测', marker='s', linestyle='-.')
    plt.title('股票价格预测（未来7日）', fontsize=14)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('收盘价', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stock_prediction.png', dpi=300)
    
    # ===== 保存预测结果 =====
    pred_df = pd.DataFrame({
        '日期': future_dates.strftime('%Y-%m-%d'),
        'SVR预测价格': np.round(svr_pred, 2),
        'ARIMA预测价格': np.round(arima_pred, 2)
    })
    pred_df.to_csv('7日股价预测.csv', index=False, encoding='utf-8-sig')
    
    print("\n===== 运行结果 =====")
    print("1. 预测图表已保存至 stock_prediction.png")
    print("2. 详细预测数据已保存至 7日股价预测.csv")
    print(pred_df.to_string(index=False))

if __name__ == "__main__":
    # 使用正斜杠并修正目录名拼写
    main('e:/finance sicence/daily_price.csv')