import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.title("株価可視化・分析アプリ")

period = st.selectbox("期間を選択", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

col1, col2 = st.columns(2)

with col1:
    st.subheader("銘柄1")
    ticker1 = st.text_input("銘柄コードを入力（例：AAPL, 7203.T）", key="q1")

with col2:
    st.subheader("銘柄2")
    ticker2 = st.text_input("銘柄コードを入力（例：NVDA, 6758.T）", key="q2")

tickers = [t for t in [ticker1, ticker2] if t]

COLORS = ["#00C896", "#FF4B4B", "#4B8BFF", "#FFB347"]

if tickers:
    all_data = {}
    for ticker in tickers:
        data = yf.download(ticker, period=period, auto_adjust=True)
        if not data.empty:
            all_data[ticker] = data

    if all_data:
        # サマリーカード
        st.markdown("---")
        summary_cols = st.columns(len(all_data))
        for i, (ticker, data) in enumerate(all_data.items()):
            close = data["Close"].squeeze()
            start_price = float(close.iloc[0])
            end_price = float(close.iloc[-1])
            change_pct = (end_price - start_price) / start_price * 100
            max_price = float(close.max())
            min_price = float(close.min())
            arrow = "▲" if change_pct >= 0 else "▼"
            color = "#00C896" if change_pct >= 0 else "#FF4B4B"

            returns = close.pct_change().dropna()
            volatility = float(returns.std() * (252 ** 0.5) * 100)
            drawdown = float(((close - close.cummax()) / close.cummax()).min() * 100)
            sharpe = float(returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() != 0 else 0

            with summary_cols[i]:
                st.markdown(f"""
<div style="background:#1e1e2e;padding:20px;border-radius:12px;border:1px solid #333;">
    <h3 style="color:#aaa;margin:0;">{ticker}</h3>
    <h1 style="color:white;margin:8px 0;">{end_price:,.2f}</h1>
    <h3 style="color:{color};margin:0;">{arrow} {change_pct:.2f}%</h3>
    <p style="color:#888;margin-top:12px;">最高値: {max_price:,.2f}<br>最安値: {min_price:,.2f}</p>
    <hr style="border-color:#333;margin:12px 0;">
    <p style="color:#aaa;margin:0;">📊 ボラティリティ: <span style="color:white;">{volatility:.1f}%</span></p>
    <p style="color:#aaa;margin:0;">📉 最大ドローダウン: <span style="color:#FF4B4B;">{drawdown:.1f}%</span></p>
    <p style="color:#aaa;margin:0;">⚡ シャープレシオ: <span style="color:#00C896;">{sharpe:.2f}</span></p>
</div>
""", unsafe_allow_html=True)

        st.markdown("---")

        # 比較グラフ
        fig = go.Figure()
        for i, (ticker, data) in enumerate(all_data.items()):
            close = data["Close"].squeeze()
            normalized = (close / close.iloc[0]) * 100
            fig.add_trace(go.Scatter(
                x=data.index, y=normalized, name=ticker,
                line=dict(color=COLORS[i % len(COLORS)], width=2)
            ))
        fig.update_layout(
            title="株価推移比較（開始時=100）",
            xaxis_title="日付",
            yaxis_title="相対価格（%）",
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#333"),
            yaxis=dict(gridcolor="#333")
        )
        st.plotly_chart(fig)

        # 急騰・急落アラート
        st.subheader("🚨 急騰・急落アラート")
        alert_cols = st.columns(len(all_data))
        for i, (ticker, data) in enumerate(all_data.items()):
            close = data["Close"].squeeze()
            daily_returns = close.pct_change().dropna()
            threshold = 0.03

            surges = daily_returns[daily_returns >= threshold]
            drops = daily_returns[daily_returns <= -threshold]

            with alert_cols[i]:
                st.markdown(f"**{ticker}**")
                if surges.empty and drops.empty:
                    st.success("この期間に急騰・急落はありませんでした")
                else:
                    for date, val in surges.items():
                        st.markdown(f"""
<div style="background:#0d2b1e;padding:10px;border-radius:8px;border-left:4px solid #00C896;margin-bottom:8px;">
    📈 <span style="color:#00C896;font-weight:bold;">急騰 +{val*100:.1f}%</span><br>
    <span style="color:#888;">{date.strftime('%Y-%m-%d')}</span>
</div>
""", unsafe_allow_html=True)
                    for date, val in drops.items():
                        st.markdown(f"""
<div style="background:#2b0d0d;padding:10px;border-radius:8px;border-left:4px solid #FF4B4B;margin-bottom:8px;">
    📉 <span style="color:#FF4B4B;font-weight:bold;">急落 {val*100:.1f}%</span><br>
    <span style="color:#888;">{date.strftime('%Y-%m-%d')}</span>
</div>
""", unsafe_allow_html=True)

        # 株価予測モデル
        st.subheader("🔮 株価予測モデル（線形回帰）")
        pred_cols = st.columns(len(all_data))
        for i, (ticker, data) in enumerate(all_data.items()):
            close = data["Close"].squeeze().reset_index(drop=True)
            X = np.arange(len(close)).reshape(-1, 1)
            y = close.values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            future_days = 30
            last_day = len(close)
            future_X = np.arange(last_day, last_day + future_days).reshape(-1, 1)
            future_pred = model.predict(future_X)
            future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1, freq="B")[1:]

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=data.index, y=close, name="実績", line=dict(color=COLORS[i % len(COLORS)], width=2)))
            fig_pred.add_trace(go.Scatter(x=future_dates, y=future_pred, name="予測", line=dict(color="#FFB347", width=2, dash="dash")))
            fig_pred.update_layout(
                title=f"{ticker}の株価予測（30日）",
                xaxis_title="日付",
                yaxis_title="価格",
                plot_bgcolor="#1e1e2e",
                paper_bgcolor="#1e1e2e",
                font=dict(color="white"),
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(gridcolor="#333")
            )

            with pred_cols[i]:
                st.plotly_chart(fig_pred, use_container_width=True)
                st.markdown(f"""
<div style="background:#1e1e2e;padding:16px;border-radius:12px;border:1px solid #333;">
    <p style="color:#aaa;margin:0;">📏 平均絶対誤差 (MAE): <span style="color:white;">{mae:.2f}</span></p>
    <p style="color:#aaa;margin:0;">📐 決定係数 (R²): <span style="color:#00C896;">{r2:.2f}</span></p>
</div>
""", unsafe_allow_html=True)

        # 移動平均（単一銘柄のみ）
        if len(tickers) == 1:
            ticker = tickers[0]
            data = all_data[ticker]
            data["MA25"] = data["Close"].rolling(25).mean()
            data["MA75"] = data["Close"].rolling(75).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data["Close"].squeeze(), name="終値", line=dict(color=COLORS[0], width=2)))
            fig2.add_trace(go.Scatter(x=data.index, y=data["MA25"].squeeze(), name="25日移動平均", line=dict(color=COLORS[1], width=1.5)))
            fig2.add_trace(go.Scatter(x=data.index, y=data["MA75"].squeeze(), name="75日移動平均", line=dict(color=COLORS[2], width=1.5)))
            fig2.update_layout(
                title=f"{ticker}の移動平均",
                xaxis_title="日付",
                yaxis_title="価格",
                plot_bgcolor="#1e1e2e",
                paper_bgcolor="#1e1e2e",
                font=dict(color="white"),
                xaxis=dict(gridcolor="#333"),
                yaxis=dict(gridcolor="#333")
            )
            st.plotly_chart(fig2)

            st.subheader("基本統計")
            st.dataframe(data["Close"].describe())