# 🎓 ULK Fees Assistant (Streamlit Chatbot)

A chatbot for **Kigali Independent University ULK** that answers questions about the
**2026–2027 fees structure (Year Ones)**: tuition for every programme, campus and schedule
differences, non-tuition fees, graduation fees, bank accounts, and contacts.

## 📁 Project structure

```
ulk-chatbot/
├── app.py                      # Streamlit chat UI
├── chatbot_engine.py           # Matching, conversation memory, answer builder
├── knowledge_base.json         # ⭐ The bot's memory: all data from the fees PDF
├── requirements.txt
├── assets/ulk_logo.png
└── .streamlit/
    ├── config.toml             # Theme (ULK colours)
    └── secrets.toml.example    # Optional AI key template
```

## 🧠 How the memory works

* **`knowledge_base.json`** is the only source of answers. It contains:
  `school` (contacts), `general_notes`, `programs` (49 programmes with duration and fees),
  `graduation_fees`, `non_tuition_fees` (38 items), `bank_accounts`, `intents` (greetings, thanks,
  help), `faqs`, and `fallback`.
* **Conversation memory** (per user session) remembers the last programme, campus and schedule, so
  follow-ups work:
  > *"Software Engineering fees"* → *"and in Gisenyi?"* → *"evening?"* → *"what about master's?"*
* Users can **💾 Save chat** as a JSON file from the sidebar.

### Updating fees next year
Edit `knowledge_base.json` only. No code changes are needed. To make the bot recognise new
words, add them to a programme's `"aliases"` list.

## ▶️ Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🚀 Deploy on Streamlit Community Cloud (free)

1. Create a **GitHub repository** and upload all files in this folder. Keep the same structure,
   including the `.streamlit/` and `assets/` folders.
2. Go to **https://share.streamlit.io**, sign in with GitHub and click **"Create app"**.
3. Choose your repo and branch, and set **Main file path** = `app.py`.
4. *(Optional)* Open **Advanced settings → Secrets** and paste the contents of
   `.streamlit/secrets.toml.example` with your key to turn on ✨ AI-enhanced answers.
5. Click **Deploy**. You'll get a link like `https://ulk-fees.streamlit.app`.

### Two answer modes
| Mode | Needs a key? | Description |
|---|---|---|
| 📘 Knowledge base (default) | No | Fast, free, accurate answers built straight from the JSON |
| ✨ AI-enhanced | Yes (OpenAI, Groq, Gemini…) | More natural replies (also in Kinyarwanda/French), still limited to the JSON data |

If the AI service fails, the bot automatically falls back to knowledge-base mode.

## ⚠️ Data notes (from the source PDF)
* **School of Science & Technology, Gisenyi:** the 200,000 Rwf/term line is labelled *"Day students"*
  in the PDF. It is stored as the **Evening** rate, following the pattern of all the other tables. Please confirm.
* **Memorant registration, row 2** (145,000 Rwf) reads *"Undergraduate with retakes"*. It probably
  means *Master's with retakes*. Please confirm.
