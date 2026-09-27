import time
import requests
import streamlit as st

# ====== ⚙️ 1. 全域配置與後端端點宣告 ======
BACKEND_URL = "http://127.0.0.1:8001"  # 💡 部署到 Docker 時可改為 http://backend:8000

st.set_page_config(
    page_title="FinQA 金融保險高階雙防禦 RAG 系統",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== 💾 2. 初始化 Streamlit 記憶體狀態機 (Session State) ======
if "messages" not in st.session_state:
    st.session_state.messages = []  # 儲存 ChatGPT 般的對話歷史紀錄

if "upload_tasks" not in st.session_state:
    st.session_state.upload_tasks = []  # 記憶體對照表：記錄所有派發出去的 Ingestion 任務狀態

# ====== 🎛️ 3. 左側側邊欄：全局業務切換控制中樞 ======
with st.sidebar:
    st.title("🏦 FinQA RAG 系統")
    st.markdown("---")
    st.subheader("📁 選擇當前業務範疇")
    
    # 強制限制前端只能選擇 insurance 或 finance，與後端 Pydantic Enum 完美對齊
    category = st.selectbox(
        "請選擇您要詢問的業務範疇：",
        options=["insurance", "finance"],
        format_func=lambda x: "🛡️ 保險業務 (Insurance)" if x == "insurance" else "📊 財務報告 (Finance)",
        key="global_category"
    )
    
    st.markdown("---")
    st.caption("⚡ **系統技術特點：**")
    st.caption("• 雙業務物理隔離儲存 (硬碟/庫內)")
    st.caption("• Redis Vector 語意快取加速")
    st.caption("• Cross-Encoder Reranker 文件重排")
    st.caption("• vLLM 推理服務")

# ====== 🥞 4. 頂部雙標籤設計 (Tabs Layout) ======
tab_qa, tab_upload = st.tabs(["💬 AI 智能語意問答", "📤 文件上傳與進度監控"])

# =========================================================
# 🔥 區塊 A：AI 智能語意問答頁面 (Tab 1)
# =========================================================
with tab_qa:
    st.header("💬 金融保險 AI 智能專家問答")
    st.markdown(f"目前對話範疇：`{category.upper()}`")
    
    # A-1. 渲染歷史對話訊息
    for msg in st.session_state.messages:
        # 只顯示符合當前所選類別的對話，切換類別時自動過濾歷史
        if msg.get("category") == category:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # 如果有來源憑據，以摺疊面板（Expander）秀出
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("🔍 資料引用處 "):
                        for src in msg["sources"]:
                            st.markdown(f"**[{src['index']}] 原始檔案：** `{src['file_name']}` | **章節標題：** `{src['header']}`")

    # A-2. 接收使用者即時提問
    if prompt := st.chat_input("請輸入您想查詢的保單條款或財務財報問題..."):
        # 先將使用者問題渲染至畫面
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 寫入歷史紀錄
        st.session_state.messages.append({"role": "user", "content": prompt, "category": category})
        
        # 呼叫後端 RAG 問答端點
        with st.chat_message("assistant"):
          with st.spinner("🔍 正在進行快取比對與檢索中..."):
            try:
              payload = {"question": prompt, "category": category}
              response = requests.post(f"{BACKEND_URL}/query", json=payload)
              
              if response.status_code == 200:
                  data = response.json()
                  answer = data["answer"]
                  sources = data["sources"]
                  is_cached = data["cached"]
                                                            
                  st.markdown(answer)
                  
                  # 即時渲染來源憑據
                  if sources:
                    with st.expander("🔍 資料引用處 "):
                      for src in sources:
                        st.markdown(f"**[{src['frontend_index']}] 原始檔案：** `{src['file_name']}` | **章節標題：** `{src['header']}`")
                  
                  # 儲存助理回覆至歷史
                  st.session_state.messages.append({
                      "role": "assistant",
                      "content": answer,
                      "sources": sources,
                      "category": category
                  })
              else:
                  st.error(f"❌ 後端服務器生成失敗 (HTTP {response.status_code}): {response.text}")
            except Exception as e:
              st.error(f"❌ 無法連線至後端微服務中心: {str(e)}")

# =========================================================
# 🔥 區塊 B：文件上傳與管線監控頁面 (Tab 2)
# =========================================================
with tab_upload:
    st.header("📤 文件檔案上傳區塊 ")
    st.markdown(f"支援最多 5 個檔案同時拖拉上傳。")
    
    
    # 限制單一檔案由前端提示，元件支援 accept_multiple_files=True 達成使用者多選體驗
    uploaded_files = st.file_uploader(
        f"請選擇要指派至 [{category.upper()}] 範疇的 PDF 報告檔案（單一檔案限制 50MB 內）：",
        type=["pdf"],
        accept_multiple_files=True
    )
    # 限制一次只能上傳 n 個 PDF
    
    is_over_limit = len(uploaded_files) > 5 if uploaded_files else False
    
    if is_over_limit:
        # 如果超過 n 個，在畫面上跳出嚴厲的警告，並顯示目前上傳了幾個
        st.error(f"🚨 一次最多只能上傳 **5** 個檔案。")
    
    # 按鈕
    if st.button("🚀 開始將檔案寫入資料庫", type="primary"):
      if not uploaded_files:
          st.warning("請先選取至少一份 PDF 文件後再行點擊執行。")
      elif is_over_limit:
          # 再次安全攔截，確保絕對不會觸發下方的 requests.post 迴圈
          st.error(f"🚨 一次最多只能上傳 **5** 個檔案。")
      else:
        # 💡 完美實現你所說的：前端 Loop 循環發送獨立的 HTTP POST 請求，防止集體卡死
        for file in uploaded_files:
          with st.spinner(f"正在傳輸檔案並建立任務: {file.name}..."):
            try:
              files_payload = {"file": (file.name, file.getvalue(), "application/pdf")}
              data_payload = {"category": category}
              
              # 呼叫後端檔案 Upload 接口 (每次只上傳一個檔案)
              res = requests.post(f"{BACKEND_URL}/upload", files=files_payload, data=data_payload)
              
              if res.status_code == 202:
                  upload_data = res.json()
                  # 將 task_id 寫入記憶體對照表中
                  st.session_state.upload_tasks.append({
                      "task_id": upload_data["task_id"],
                      "file_name": file.name,
                      "category": category,
                      "status": "PENDING",
                      "stage": "排隊佇列中",
                      "progress": 0.1
                  })
                  st.success(f"✅ 檔案 `{file.name}` 成功推入排隊隊伍！Task ID 註冊成功。")
              elif res.status_code == 413:
                  st.error(f"❌ 檔案 `{file.name}` 遭拒絕：檔案大小超過 50MB 限制！")
              else:
                  st.error(f"❌ 檔案 `{file.name}` 上傳失敗: {res.text}")
            except Exception as e:
                st.error(f"❌ 傳輸 `{file.name}` 時發生連線錯誤: {str(e)}")
        
        st.toast("所有檔案已全數派發至 Celery 後台！請在下方查看動態進度條。")

    # ====== 📊 B-2. 任務生命週期動態儀表板 (進度條 Polling 區塊) ======
    st.markdown("---")
    st.subheader("📊 記憶體任務進度動態監控儀表板")
    
    if not st.session_state.upload_tasks:
        st.info("目前暫無進行中或排隊中的 PDF寫入任務。")
    else:
        # 提供一個按鈕讓使用者可以手動強制重新整理，或是由下面輪詢自然更新
        if st.button("🔄 手動刷新所有任務狀態"):
            st.rerun()
            
        # 遍歷目前儲存在對照表裡的所有背景任務
        for task in st.session_state.upload_tasks:
          # 唯有當任務不是最終狀態時，才去戳後端 API，節省效能
          if task["status"] not in ["SUCCESS", "FAILURE"]:
              try:
                # 💡 輪詢我們設計的 /task/{task_id} 大門
                status_res = requests.get(f"{BACKEND_URL}/task/{task['task_id']}")
                if status_res.status_code == 200:
                    s_data = status_res.json()
                    task["status"] = s_data["status"]
                    task["stage"] = s_data["current_stage"]
                    
                    # 依據後端更新的 Checkpoint 狀態，動態畫出不同的前端進度條百分比
                    if s_data["status"] == "PENDING":
                        task["progress"] = 0
                    elif s_data["status"] == "PROCESSING":
                        if "OCR" in s_data["current_stage"]:
                            task["progress"] = 0.2
                        elif "Parent" in s_data["current_stage"]:
                            task["progress"] = 0.5
                        else:
                            task["progress"] = 0.8
                    elif s_data["status"] == "SUCCESS":
                        task["progress"] = 1.0
                    elif s_data["status"] == "FAILURE":
                        task["progress"] = 1.0
              except Exception:
                  task["stage"] = "無法取得後端監控連線"

          # 🛠️ 在前端畫出極具工業感的元件排版
          col_info, col_bar = st.columns([2, 3])
          with col_info:
              # 依據狀態給予不同的 icon 符號
              icon = "⏳" if task["status"] in ["PENDING", "PROCESSING"] else ("🟢" if task["status"] == "SUCCESS" else "🚨")
              st.markdown(f"{icon} **[{task['category'].upper()}]** `{task['file_name']}`")
              st.caption(f"任務標籤：`{task['task_id']}` | 當前節點：*{task['stage']}*")
              
          with col_bar:
              if task["status"] == "SUCCESS":
                  st.progress(1.0)
                  st.success("🎉 檔案寫入成功！")
              elif task["status"] == "FAILURE":
                  st.progress(1.0)
                  st.error("🚨 寫入失敗！檔案已進入失敗任務區（DLQ）等待複查。")
              else:
                  # 進行中的任務，畫出動態進度條
                  st.progress(task["progress"])
          st.markdown("---")

