import streamlit as st
import asyncio
import logging
import json
from management_panel import get_currency_pairs, add_currency_pair, remove_currency_pair
from telegram_notifier import TelegramNotifier
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Forex Trading Management Panel")

# JSON file functions
def load_signals():
    try:
        with open('signals.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_signals(signals):
    with open('signals.json', 'w') as f:
        json.dump(signals, f)

# Initialize signals
if 'signals' not in st.session_state:
    st.session_state.signals = load_signals()

# Add signal function
def add_signal(pair, action, price, stop_loss, take_profit):
    logger.debug(f"Adding signal: pair={pair}, action={action}, price={price}, stop_loss={stop_loss}, take_profit={take_profit}")
    try:
        if pair not in st.session_state.signals:
            st.session_state.signals[pair] = []
        
        signal = {
            'action': action,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'Open'
        }
        st.session_state.signals[pair].append(signal)
        save_signals(st.session_state.signals)
        logger.info(f"Signal added successfully for {pair}")
        return True
    except Exception as e:
        logger.error(f"Error adding signal: {str(e)}")
        return False

# Currency Pair Management
st.header("Currency Pair Management")

# Add new currency pair
new_pair = st.text_input("Add new currency pair (e.g., GBPUSD)")
if st.button("Add Pair"):
    if add_currency_pair(new_pair):
        st.success(f"Added {new_pair} to the list.")
    else:
        st.warning(f"{new_pair} is already in the list.")

# Remove currency pair
pair_to_remove = st.selectbox("Select pair to remove", get_currency_pairs())
if st.button("Remove Pair"):
    if remove_currency_pair(pair_to_remove):
        st.success(f"Removed {pair_to_remove} from the list.")
    else:
        st.warning(f"Failed to remove {pair_to_remove}.")

# Display current currency pairs
st.subheader("Current Currency Pairs")
st.write(get_currency_pairs())

# Telegram Settings
st.header("Telegram Settings")
new_token = st.text_input("Telegram Bot Token", value=st.secrets.get("TELEGRAM_BOT_TOKEN", ""))
new_chat_id = st.text_input("Telegram Chat ID", value=st.secrets.get("TELEGRAM_CHAT_ID", ""))

def update_secrets(token, chat_id):
    secrets_path = ".streamlit/secrets.toml"
    with open(secrets_path, "r") as f:
        secrets_content = f.read()
    
    secrets_content = secrets_content.replace(f'TELEGRAM_BOT_TOKEN = "{st.secrets.get("TELEGRAM_BOT_TOKEN", "")}"', f'TELEGRAM_BOT_TOKEN = "{token}"')
    secrets_content = secrets_content.replace(f'TELEGRAM_CHAT_ID = "{st.secrets.get("TELEGRAM_CHAT_ID", "")}"', f'TELEGRAM_CHAT_ID = "{chat_id}"')
    
    with open(secrets_path, "w") as f:
        f.write(secrets_content)

if st.button("Update Telegram Settings"):
    update_secrets(new_token, new_chat_id)
    st.success("Telegram settings updated successfully.")

# Send Telegram Message
st.header("Send Telegram Message")
message = st.text_area("Enter your message")
if st.button("Send Message"):
    try:
        logger.info("Attempting to send Telegram message")
        notifier = TelegramNotifier(new_token)
        asyncio.run(notifier.send_message(new_chat_id, message))
        logger.info("Telegram message sent successfully")
        st.success("Message sent successfully.")
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}")
        logger.exception("Detailed error information:")
        st.error(f"Failed to send message: {str(e)}")

# Trading Signals
st.header("Trading Signals")
signals = st.session_state.signals

# Calculate accuracy rate
total_signals = sum(len(pair_signals) for pair_signals in signals.values())
successful_signals = sum(1 for pair_signals in signals.values() for signal in pair_signals if signal['status'] in ['TP1 Hit', 'TP2 Hit', 'TP3 Hit'])
accuracy_rate = (successful_signals / total_signals) * 100 if total_signals > 0 else 0

st.subheader(f"Accuracy Rate: {accuracy_rate:.2f}%")

# Display signals
for pair, pair_signals in signals.items():
    st.subheader(f"Signals for {pair}")
    for signal in pair_signals:
        st.write(f"Action: {signal['action']}, Price: {signal['price']:.5f}, Status: {signal['status']}")

# Add a button to go back to the main page
if st.button("Go to Main Page"):
    st.switch_page("main.py")
