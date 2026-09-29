import os
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from google import genai

# -----------------------------------------------------------------------------
# 1. ページ構成 & デザイン
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Stock Advisor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 個人専用 AI株式スクリーナー & アドバイザー")
st.caption("リアルタイム市場データ × テクニカル分析 × 最新AIによるパーソナル投資支援")

# -----------------------------------------------------------------------------
# 2. テクニカル分析 & スクリーニング関数
# -----------------------------------------------------------------------------
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI（相対力指数）を算出"""
    if len(prices) < period + 1:
        return np.nan
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

@st.cache_data(ttl=300)
def fetch_ticker_data(symbol: str):
    """銘柄の株価・RSI・売上成長率・時価総額を取得（5分間キャッシュ）"""
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="6mo")
        if hist.empty:
            return None
        
        current_price = hist["Close"].iloc[-1]
        prev_price = hist["Close"].iloc[-2] if len(hist) > 1 else current_price
        pct_change = ((current_price - prev_price) / prev_price) * 100
        rsi = calculate_rsi(hist["Close"], period=14)
        
        info = t.info or {}
        rev_growth = info.get("revenueGrowth", None)
        
        return {
            "symbol": symbol,
            "name": info.get("shortName", symbol),
            "price": current_price,
            "change": pct_change,
            "rsi": rsi,
            "rev_growth": (rev_growth * 100) if rev_growth is not None else None,
            "market_cap": info.get("marketCap", 0),
            "history": hist
        }
    except Exception:
        return None

# -----------------------------------------------------------------------------
# 3. サイドバー: あなたの投資プロフィール設定
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ ユーザー設定")
    
    api_key = st.text_input(
        "Gemini APIキー",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Google AI Studioで無料発行したAPIキーを貼り付けます"
    )
    
    st.divider()
    st.subheader("👤 ライフプラン & 投資枠")
    age = st.number_input("年齢", value=48, min_value=18, max_value=90)
    family = st.text_input("家族構成", value="妻、子供3人（9歳、7歳、5歳）")
    income = st.number_input("年収（手取り概算 / 万円）", value=700, step=50)
    monthly_budget = st.number_input("毎月の投資可能枠（円）", value=100000, step=10000)
    
    st.divider()
    st.subheader("🔍 監視対象銘柄")
    st.caption("カンマ区切りで入力（米国株: NVDA, TSLA / 日本株: 7203.T, 6758.T など）")
    default_list = "NVDA, AAPL, MSFT, TSLA, PLTR, 7203.T"
    input_tickers = st.text_area("ティッカーシンボル", value=default_list, height=90)
    ticker_list = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]

# -----------------------------------------------------------------------------
# 4. 為替 & スクリーニング結果のリアルタイム表示
# -----------------------------------------------------------------------------
col_fx, col_date = st.columns([1, 2])
with col_fx:
    try:
        usd_jpy_hist = yf.Ticker("USDJPY=X").history(period="1d")
        rate = usd_jpy_hist["Close"].iloc[-1]
        st.metric("為替レート (USD / JPY)", f"{rate:.2f} 円")
    except Exception:
        st.metric("為替レート (USD / JPY)", "取得不可")

st.divider()
st.subheader("🎯 銘柄スクリーニング（押し目シグナル ＆ 成長性）")

results = []
histories = {}
for sym in ticker_list:
    data = fetch_ticker_data(sym)
    if data:
        results.append(data)
        histories[sym] = data["history"]

if results:
    df = pd.DataFrame(results)
    
    # 判定フラグ付与
    df["押し目サイン"] = df["rsi"].apply(lambda x: "🟢 押し目買い検討 (RSI≤35)" if pd.notnull(x) and x <= 35 else "ー")
    df["成長性判定"] = df["rev_growth"].apply(lambda x: "🔥 テンバガー候補 (売上+30%超)" if pd.notnull(x) and x >= 30 else "ー")
    
    table_df = pd.DataFrame({
        "銘柄": df["symbol"],
        "企業名": df["name"],
        "最新株価": df["price"].apply(lambda x: f"{x:,.2f}"),
        "前日比": df["change"].apply(lambda x: f"{x:+.2f}%"),
        "RSI (14日)": df["rsi"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "N/A"),
        "押し目判定": df["押し目サイン"],
        "売上成長率": df["rev_growth"].apply(lambda x: f"{x:+.1f}%" if pd.notnull(x) else "N/A"),
        "テンバガー条件": df["成長性判定"],
    })
    st.dataframe(table_df, use_container_width=True)
else:
    st.warning("銘柄データが取得できませんでした。ティッカー名を確認してください。")

# -----------------------------------------------------------------------------
# 5. 個別株価チャート描画
# -----------------------------------------------------------------------------
if ticker_list:
    selected_chart = st.selectbox("詳細チャートを表示する銘柄を選択", ticker_list)
    if selected_chart in histories and not histories[selected_chart].empty:
        hist_df = histories[selected_chart]
        fig = go.Figure(data=[go.Candlestick(
            x=hist_df.index,
            open=hist_df['Open'],
            high=hist_df['High'],
            low=hist_df['Low'],
            close=hist_df['Close'],
            name="株価"
        )])
        fig.update_layout(
            title=f"{selected_chart} 直近6ヶ月の株価推移",
            xaxis_title="日付",
            yaxis_title="価格",
            xaxis_rangeslider_visible=False,
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 6. AIパーソナル診断・総合アドバイス
# -----------------------------------------------------------------------------
st.divider()
st.subheader("🤖 AIパーソナル総合診断")

if st.button("現在の市場環境とプロフィールから診断を生成", type="primary"):
    if not api_key:
        st.error("分析を実行するには、左のサイドバーにGemini APIキーを入力してください。")
    elif not results:
        st.error("スクリーニング対象のデータが存在しません。")
    else:
        with st.spinner("世界情勢・為替・スクリーニング結果・ライフプランを統合分析中..."):
            try:
                # 最新SDK (google-genai) でクライアント初期化
                client = genai.Client(api_key=api_key)
                
                # 銘柄要約テキスト
                stock_summary = ""
                for r in results:
                    rsi_val = f"{r['rsi']:.1f}" if pd.notnull(r['rsi']) else 'N/A'
                    growth_val = f"{r['rev_growth']:.1f}%" if pd.notnull(r['rev_growth']) else 'N/A'
                    stock_summary += f"- {r['symbol']}: 現在価格={r['price']:.2f}, RSI={rsi_val}, 売上成長率={growth_val}\n"

                prompt = f"""
あなたは客観的で極めて優秀なフィナンシャルアドバイザー兼株式アナリストです。
以下の登録者プロフィールと最新市場データを踏まえ、具体的で現実的なアドバイスレポートを作成してください。

### 【登録者プロフィール】
* 年齢: {age}歳
* 家族構成: {family}（子供3人の将来の教育費発生タイミングを考慮）
* 年収（手取り概算）: {income}万円
* 毎月の投資可能余力: {monthly_budget:,}円

### 【リアルタイム市場スナップショット】
* 為替 (USD/JPY): {rate:.2f} 円
* 監視対象のスクリーニング状況:
{stock_summary}

---
### 【レポート必須項目】
1. **ライフプランを踏まえたポートフォリオ診断**
   （子供3人の教育費ピーク期や定年退職までの年数を意識した、安全資産とリスク資産の最適配分）
2. **為替・世界情勢（マクロ環境）の現在地**
   （日米金利差や地政学リスクが、日本株・米国株に与える影響と注意点）
3. **スクリーニング結果の判定（押し目買いチェック）**
   （RSIが低い銘柄について、単なる割安・反発狙いができる局面か、事業リスクによる下落かの切り分け）
4. **5年後にテンバガー（10倍株）を狙うためのスクリーニング・戦略**
   （売上高30%超の銘柄が持つポテンシャルと、高PER銘柄を保有する際のボラティリティ対策）
5. **今月〜次期のアクションプラン**
   （登録者が無理なく実行できる具体的な積立・買い増し配分の指針）

※ 免責事項: 「本レポートは情報提供のみを目的としており、個別株式の売買を推奨または保証するものではありません。」と明記してください。
"""
                # 2026年最新の安定推論モデル (gemini-2.5-flash) を指定
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"AI診断中にエラーが発生しました: {e}")
