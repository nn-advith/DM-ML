mkdir -p ~/.streamlit/
echo "[theme]
base='light'
primaryColor='#ff1490'
secondaryBackgroundColor='#f9a0dc'
textColor='#000000'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml