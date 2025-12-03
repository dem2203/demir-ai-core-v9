import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.brain.optimizer import GeneticOptimizer
import time

def render_optimizer_page():
    st.title("🧬 DEMIR AI - Strategy Optimizer")
    st.markdown("Bu modül, **Genetik Algoritma** kullanarak mevcut piyasa koşulları (Rejim) için en uygun parametreleri evrimsel süreçle bulur.")
    
    # --- Sidebar Ayarları ---
    with st.sidebar:
        st.header("⚙️ Optimizasyon Ayarları")
        symbol = st.selectbox("Parite", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
        timeframe = st.selectbox("Zaman Dilimi", ["15m", "1h", "4h", "1d"], index=2)
        
        st.divider()
        
        generations = st.slider("Jenerasyon Sayısı (Evrim)", min_value=3, max_value=50, value=10)
        population = st.slider("Popülasyon Büyüklüğü", min_value=10, max_value=100, value=20)
        
        start_btn = st.button("🚀 Optimizasyonu Başlat", use_container_width=True)

    # --- Ana Alan ---
    if start_btn:
        optimizer = GeneticOptimizer(symbol=symbol, timeframe=timeframe, population_size=population)
        
        # Durum Göstergeleri
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Metrikler için placeholder
        col1, col2 = st.columns(2)
        metric_best = col1.empty()
        metric_avg = col2.empty()
        
        # Grafik için placeholder
        chart_placeholder = st.empty()
        
        history_data = []

        def update_ui(gen, total_gen, best, avg):
            progress = gen / total_gen
            progress_bar.progress(progress)
            status_text.info(f"Evrimleştiriliyor... Jenerasyon: {gen}/{total_gen}")
            
            metric_best.metric("En İyi Getiri (ROI)", f"%{best:.2f}")
            metric_avg.metric("Ortalama Getiri", f"%{avg:.2f}")
            
            # Canlı Grafik Güncelleme
            history_data.append({"Generation": gen, "Best Fitness": best, "Avg Fitness": avg})
            df_hist = pd.DataFrame(history_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_hist["Generation"], y=df_hist["Best Fitness"], mode='lines+markers', name='En İyi Skor', line=dict(color='#00ff00')))
            fig.add_trace(go.Scatter(x=df_hist["Generation"], y=df_hist["Avg Fitness"], mode='lines', name='Ortalama', line=dict(color='#ffff00', dash='dash')))
            fig.update_layout(title='Evrimsel Gelişim Eğrisi', xaxis_title='Jenerasyon', yaxis_title='Net Kar (%)', template='plotly_dark')
            chart_placeholder.plotly_chart(fig, use_container_width=True)

        try:
            with st.spinner("Binance'den Gerçek Veri Çekiliyor ve Doğrulanıyor..."):
                best_params, best_score, history = optimizer.optimize(generations=generations, callback=update_ui)
            
            st.success("✅ Optimizasyon Tamamlandı!")
            status_text.empty()
            
            # Sonuçları Göster
            st.subheader("🏆 Kazanan Genom (Parametre Seti)")
            
            # Parametreleri güzel bir JSON veya tablo olarak göster
            st.json(best_params)
            
            # Detaylı Analiz
            st.info(f"Bu parametre seti ile {timeframe} grafiğinde son 500 mumda **%{best_score:.2f}** net kar elde edildi.")
            
        except Exception as e:
            st.error(f"Bir hata oluştu: {str(e)}")

    else:
        st.info("Sol menüden ayarları yapıp 'Optimizasyonu Başlat' butonuna basın.")
        st.markdown("""
        ### 🔍 Nasıl Çalışır?
        1. **Gerçek Veri:** Binance API üzerinden canlı mum verileri çekilir.
        2. **Doğal Seçilim:** Rastgele parametrelerle (RSI uzunluğu, Stop Loss oranı vb.) binlerce strateji oluşturulur.
        3. **Backtest:** Her strateji gerçek veride test edilir.
        4. **Evrim:** En çok kazandıran stratejiler "çiftleştirilir" ve mutasyona uğratılarak daha iyi nesiller üretilir.
        """)
