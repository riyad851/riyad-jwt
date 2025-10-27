#MD HAFIJUR ROHMAN
#CONTRACT TELEGRAM :@hafijur968
#DO NOT REMOVE THIS
import os
import json
import aiohttp
import asyncio
import time
import logging
import traceback
import re
from datetime import datetime, timedelta, timezone
from html import escape
from collections import defaultdict

# Telegram Bot Library Imports
from telegram import Update, InputFile, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackContext,
    ContextTypes,
)
from telegram.constants import ParseMode
from telegram.error import TelegramError, Forbidden, BadRequest

# Environment Variable Loading
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

# Essential: Get Bot Token
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN') # Prefer .env
if not TOKEN:
    # Fallback - Replace with your actual bot token if not using .env
    TOKEN = "8393121728:AAE7bVzU4riv8hGHcQqgZhLTmj1c7qMjbF8"

# API Configuration
API_BASE_URL = os.getenv('JWT_API_URL', 'https://hafijur-jwt-gen.vercel.app/token')
API_KEY = os.getenv('JWT_API_KEY', '')

# Bot Settings
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 5 * 1024 * 1024))  # 5MB default
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', 10))
DEVELOPER_USERNAME = '@riyadalhasan10'  # Added developer username

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'bot_data')
TEMP_DIR = os.path.join(DATA_DIR, 'temp_files')

# --- Logging Setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.vendor.ptb_urllib3.urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def format_time(seconds: float) -> str:
    """Formats seconds into a human-readable HH:MM:SS string."""
    if seconds is None or seconds < 0: return "N/A"
    try:
        seconds_int = int(seconds)
        if seconds_int < 60:
            return f"{seconds_int}s" if seconds_int >= 0 else "0s"
        delta = timedelta(seconds=seconds_int)
        total_seconds = delta.total_seconds()
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if hours > 0: parts.append(f"{int(hours)}h")
        if minutes > 0 or (hours > 0 and seconds > 0): parts.append(f"{int(minutes)}m")
        if seconds > 0 or (not parts and total_seconds >=0): parts.append(f"{int(seconds)}s")

        if not parts: return "0s"

        return " ".join(parts).strip()

    except (OverflowError, ValueError):
        return "Infinity"
    except Exception as e:
        logger.warning(f"Error formatting time {seconds}: {e}")
        return "Format Error"

def load_json_data(filepath: str, default_value=None) -> dict | list:
    """Loads JSON data from a file, returning default_value on error or if file not found."""
    if default_value is None:
        default_value = {}
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return default_value
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return default_value
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {e}")
        return default_value

def save_json_data(filepath: str, data: dict | list) -> bool:
    """Saves data to a JSON file using atomic write. Returns True on success, False on error."""
    temp_filepath = filepath + ".tmp"
    try:
        dir_name = os.path.dirname(filepath)
        if dir_name:
             os.makedirs(dir_name, exist_ok=True)

        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        os.replace(temp_filepath, filepath)
        logger.debug(f"Successfully saved data to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        return False
    finally:
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError:
                pass

# --- Command Buttons ---
COMMAND_BUTTONS_LAYOUT = [
    ["Process File 📤", "Help 🆘"],
    ["Cancel ❌"]
]
main_reply_markup = ReplyKeyboardMarkup(COMMAND_BUTTONS_LAYOUT, resize_keyboard=True, one_time_keyboard=False)

# --- Bot Command Handlers ---

async def start(update: Update, context: CallbackContext) -> None:
    """Send welcome message with buttons."""
    user = update.effective_user
    if not user: return

    username = escape(user.first_name) or "there"

    start_msg = f"👋 Hello {username}!\n\n"
    start_msg += "🚀 Welcome to the DG JWT Token Generator Bot!\n\n"
    start_msg += "📁 Send me a JSON file containing account credentials like this:\n"
    start_msg += "```json\n"
    start_msg += '[\n'
    start_msg += '    {"uid": "user1", "password": "pass1"},\n'
    start_msg += '    {"uid": "user2", "password": "pass2"}\n'
    start_msg += '    // ... more entries ...\n'
    start_msg += ']\n'
    start_msg += "```\n"
    start_msg += "✅ Successful tokens (Region summary included in message) will be saved to `jwt_token.json` AND `accounts{Region}.json` files.\n"
    start_msg += "✔️ Working accounts (UID/Pass) will be saved to `working_account.json`\n"
    start_msg += "❌ Failed/invalid entries (UID/Pass) will be saved to `lost_account.json`\n\n"
    start_msg += f"⚠️ Max file size: {MAX_FILE_SIZE / 1024 / 1024:.1f}MB\n\n"
    start_msg += f"👨‍💻 Developer: {DEVELOPER_USERNAME}\n\n"
    start_msg += "Use /help or the Help button (🆘) for more information."

    await update.message.reply_text(
        start_msg,
        reply_markup=main_reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: CallbackContext) -> None:
    """Send the detailed help message."""
    user = update.effective_user
    if user:
        pass  # User exists but no specific action needed

    help_text = (
        "🆘 *Help Center*\n\n"
        "📌 *Available Commands:*\n"
        "  `/start` - Show the main welcome message\n"
        "  `/help` - Show this help message\n"
        "  `/cancel` - Cancel the current operation\n\n"
        "📤 *How to Process Files:*\n"
        "  1. Send a JSON file formatted with UID-password pairs.\n"
        "  2. The bot processes it and returns result files.\n"
        "  3. You'll receive:\n"
        "     - `jwt_token.json` - All successful tokens\n"
        "     - `accounts{Region}.json` - Region-specific tokens\n"
        "     - `working_account.json` - Successful UID/password pairs\n"
        "     - `lost_account.json` - Failed accounts with error reasons\n\n"
        f"👨‍💻 *Developer:* {DEVELOPER_USERNAME}\n"
        "For any issues or questions, contact the developer above."
    )
    await update.message.reply_text(
        help_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=main_reply_markup,
        disable_web_page_preview=True
    )

async def cancel(update: Update, context: CallbackContext) -> None:
    """Handles the /cancel command or Cancel button, clearing pending actions."""
    user = update.effective_user
    user_id = user.id if user else "Unknown"
    
    if context.user_data.pop('waiting_for_json', None):
        logger.info(f"User {user_id} cancelled waiting for manual JSON process.")
        await update.message.reply_text(
            "Waiting for manual process file cancelled. Returning to main menu.",
            reply_markup=main_reply_markup
        )
    else:
        logger.info(f"User {user_id} used /cancel, but no active operation found.")
        await update.message.reply_text(
            "No active operation to cancel. Returning to main menu.",
            reply_markup=main_reply_markup
        )

# --- File Processing Logic ---

async def process_account(session: aiohttp.ClientSession, account: dict, semaphore: asyncio.Semaphore) -> tuple[str | None, str | None, dict | None, dict | None, str | None]:
    """
    Processes a single account via the API to get a JWT token and potentially region.
    Returns: tuple(token | None, region | None, working_account | None, lost_account | None, error_reason | None)
    """
    uid = account.get("uid")
    password = account.get("password")
    error_reason = None
    original_account_info = account.copy()

    if not uid: error_reason = "Missing 'uid'"
    elif not password: error_reason = "Missing 'password'"

    if error_reason:
        logger.debug(f"Skipping account due to validation error: {error_reason} - Account: {account}")
        lost_info = {**original_account_info, "error_reason": error_reason}
        return None, None, None, lost_info, error_reason

    uid_str = str(uid)

    async with semaphore:
        params = {'uid': uid_str, 'password': password, 'key': API_KEY}
        try:
            async with session.get(API_BASE_URL, params=params, timeout=aiohttp.ClientTimeout(total=60)) as response:
                response_text = await response.text()

                if 200 <= response.status < 300:
                    try:
                        result = json.loads(response_text)
                        if isinstance(result, dict) and result.get('token'):
                            token = result['token']
                            region = result.get('region')
                            logger.info(f"Success: Token received for UID: {uid_str} (Region: {region})")
                            return token, region, original_account_info, None, None
                        else:
                            err_msg = "API OK but invalid response format or empty token"
                            logger.warning(f"{err_msg} for UID: {uid_str}. Response: {response_text[:200]}")
                            lost_info = {**original_account_info, "error_reason": err_msg}
                            return None, None, None, lost_info, err_msg
                    except json.JSONDecodeError:
                        err_msg = f"API OK ({response.status}) but Non-JSON response"
                        logger.error(f"{err_msg} for UID: {uid_str}. Response: {response_text[:200]}")
                        lost_info = {**original_account_info, "error_reason": err_msg}
                        return None, None, None, lost_info, err_msg
                    except Exception as e:
                         err_msg = f"API OK ({response.status}) but response parsing error: {e}"
                         logger.error(f"{err_msg} for UID: {uid_str}")
                         lost_info = {**original_account_info, "error_reason": err_msg}
                         return None, None, None, lost_info, err_msg

                else:
                    error_detail = f"API Error ({response.status})"
                    try:
                        error_json = json.loads(response_text)
                        if isinstance(error_json, dict):
                            msg = error_json.get('message') or error_json.get('error') or error_json.get('detail')
                            if msg and isinstance(msg, str):
                                error_detail += f": {msg[:100]}"
                    except (json.JSONDecodeError, TypeError): pass

                    logger.warning(f"API Error for UID: {uid_str}. Status: {response.status}. Detail: {error_detail}. Raw Response: {response_text[:200]}")
                    lost_info = {**original_account_info, "error_reason": error_detail}
                    return None, None, None, lost_info, error_detail

        except asyncio.TimeoutError:
             logger.warning(f"Timeout processing API request for UID: {uid_str}")
             error_reason = "Request Timeout"
             lost_info = {**original_account_info, "error_reason": error_reason}
             return None, None, None, lost_info, error_reason
        except aiohttp.ClientConnectorError as e:
             logger.error(f"Network Connection Error processing UID {uid_str}: {e}")
             error_reason = f"Network Error: {e}"
             lost_info = {**original_account_info, "error_reason": error_reason}
             return None, None, None, lost_info, error_reason
        except aiohttp.ClientError as e:
             logger.error(f"AIOHTTP Client Error processing UID {uid_str}: {e}")
             error_reason = f"HTTP Client Error: {e}"
             lost_info = {**original_account_info, "error_reason": error_reason}
             return None, None, None, lost_info, error_reason
        except Exception as e:
             logger.error(f"Unexpected error processing UID {uid_str}: {e}")
             error_reason = f"Unexpected Processing Error: {e}"
             lost_info = {**original_account_info, "error_reason": error_reason}
             return None, None, None, lost_info, error_reason

async def handle_document(update: Update, context: CallbackContext) -> None:
    """Handle incoming JSON documents for processing."""
    user = update.effective_user
    message = update.message
    if not user or not message: return
    user_id = user.id
    chat_id = message.chat_id

    # Check if this is a button click to process file
    process_button_text = COMMAND_BUTTONS_LAYOUT[0][0]
    if message.text == process_button_text and not message.document:
        await message.reply_text(
            "Okay, please send the JSON file now for processing.\n\n"
            "Make sure it's a `.json` file containing a list like:\n"
            "```json\n"
            '[\n  {"uid": "user1", "password": "pass1"},\n  {"uid": "user2", "password": "pass2"}\n]\n'
            "```",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=ReplyKeyboardRemove()
        )
        context.user_data['waiting_for_json'] = True
        return

    was_waiting_manual = context.user_data.pop('waiting_for_json', False)
    if was_waiting_manual and not message.document:
         await message.reply_text("Looks like you sent text instead of a file. Please send the JSON file or use /cancel.", reply_markup=main_reply_markup)
         return
    elif not was_waiting_manual and not message.document:
        return

    document = message.document
    if not document: return

    # File Validation
    is_json_mime = document.mime_type and document.mime_type.lower() == 'application/json'
    has_json_extension = document.file_name and document.file_name.lower().endswith('.json')

    if not is_json_mime and not has_json_extension:
        await message.reply_text("❌ File does not appear to be a JSON file. Please ensure it has a `.json` extension or the correct `application/json` type.", reply_markup=main_reply_markup)
        return

    file_id = document.file_id
    file_name = document.file_name or f"file_{file_id}.json"

    # Check file size before download if possible
    if document.file_size and document.file_size > MAX_FILE_SIZE:
        await message.reply_text(
            f"⚠️ File is too large ({document.file_size / 1024 / 1024:.2f} MB). Max: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB.",
            reply_markup=main_reply_markup
        )
        return

    temp_file_path = os.path.join(TEMP_DIR, f'input_manual_{user_id}_{int(time.time())}.json')
    progress_message = None
    accounts_data = []

    # Download and Parse
    try:
        os.makedirs(TEMP_DIR, exist_ok=True)
        progress_message = await message.reply_text(f"⏳ Downloading `{escape(file_name)}` for processing...", parse_mode=ParseMode.MARKDOWN)

        bot_file = await context.bot.get_file(file_id)
        await bot_file.download_to_drive(temp_file_path)
        logger.info(f"User {user_id} uploaded file '{file_name}', downloaded to {temp_file_path}")

        await context.bot.edit_message_text(
            chat_id=progress_message.chat_id, message_id=progress_message.message_id,
            text=f"⏳ Downloaded `{escape(file_name)}`. Parsing JSON...", parse_mode=ParseMode.MARKDOWN
        )

        # Check size after download
        actual_size = os.path.getsize(temp_file_path)
        if actual_size > MAX_FILE_SIZE:
             raise ValueError(f"Downloaded file size ({actual_size / 1024 / 1024:.2f} MB) exceeds limit ({MAX_FILE_SIZE / 1024 / 1024:.1f} MB).")

        with open(temp_file_path, 'r', encoding='utf-8') as f:
            try:
                accounts_data = json.load(f)
            except json.JSONDecodeError as e:
                error_line_info = ""
                if hasattr(e, 'lineno') and hasattr(e, 'colno'):
                    error_line_info = f" near line {e.lineno}, column {e.colno}"
                error_msg = f"❌ Invalid JSON format in `{escape(file_name)}`{error_line_info}.\nError: `{escape(e.msg)}`.\nPlease check the file structure and syntax."
                await context.bot.edit_message_text(
                    chat_id=progress_message.chat_id, message_id=progress_message.message_id,
                    text=error_msg, parse_mode=ParseMode.MARKDOWN
                )
                return

        # Validate JSON structure
        if not isinstance(accounts_data, list):
            raise ValueError("Input JSON structure is invalid. It must be an array (a list `[...]`) of objects.")
        if accounts_data and not all(isinstance(item, dict) for item in accounts_data):
             first_bad_item = next((item for item in accounts_data if not isinstance(item, dict)), None)
             raise ValueError(f"All items inside the JSON array must be objects (`{{...}}`). Found an item that is not an object: `{escape(str(first_bad_item)[:50])}`...")

    except ValueError as e:
        logger.warning(f"Input file validation failed for user {user_id} ('{file_name}'): {e}")
        error_text = f"❌ Validation Error: {escape(str(e))}"
        if progress_message:
             await context.bot.edit_message_text(chat_id=progress_message.chat_id, message_id=progress_message.message_id, text=error_text, parse_mode=ParseMode.MARKDOWN)
        else:
             await message.reply_text(error_text, reply_markup=main_reply_markup, parse_mode=ParseMode.MARKDOWN)
        return
    except TelegramError as e:
        logger.error(f"Telegram API error during file handling for user {user_id}: {e}")
        try:
            error_text = f"⚠️ A Telegram error occurred: `{escape(str(e))}`. Please try again later."
            if progress_message:
                await context.bot.edit_message_text(chat_id=progress_message.chat_id, message_id=progress_message.message_id, text=error_text, parse_mode=ParseMode.MARKDOWN)
            else:
                 await message.reply_text(error_text, reply_markup=main_reply_markup, parse_mode=ParseMode.MARKDOWN)
        except TelegramError:
            logger.error(f"Could not inform user {user_id} about Telegram error: {e}")
        return
    except Exception as e:
        logger.error(f"Error downloading or parsing file from user {user_id}: {e}")
        error_text = f"⚠️ An unexpected error occurred while handling the file. Please try again."
        if progress_message:
            try:
                await context.bot.edit_message_text(chat_id=progress_message.chat_id, message_id=progress_message.message_id, text=error_text)
            except TelegramError:
                await message.reply_text(error_text, reply_markup=main_reply_markup)
        else:
            await message.reply_text(error_text, reply_markup=main_reply_markup)
        return
    finally:
        if os.path.exists(temp_file_path):
             try:
                 os.remove(temp_file_path)
             except OSError as e:
                 logger.warning(f"Could not remove temp input file {temp_file_path}: {e}")

    # Process Accounts
    total_count = len(accounts_data)
    if total_count == 0:
        await context.bot.edit_message_text(
            chat_id=progress_message.chat_id, message_id=progress_message.message_id,
            text="ℹ️ The provided JSON file is empty or contains no valid account objects."
        )
        return

    await context.bot.edit_message_text(
        chat_id=progress_message.chat_id, message_id=progress_message.message_id,
        text=f"🔄 *Processing {total_count} Accounts*\nInitializing API calls (max {MAX_CONCURRENT_REQUESTS} parallel)...",
        parse_mode=ParseMode.MARKDOWN
    )

    start_time = time.time()
    processed_count = 0
    successful_tokens = []
    working_accounts = []
    lost_accounts = []
    errors_summary = defaultdict(int)

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        tasks = [process_account(session, account, semaphore) for account in accounts_data]
        last_update_time = time.time()
        last_progress_text_sent = ""

        for future in asyncio.as_completed(tasks):
            try:
                token, region, working_acc, lost_acc, error_reason = await future
            except Exception as task_err:
                logger.error(f"Error retrieving result from processing task: {task_err}")
                error_msg = f"Internal task error: {task_err}"
                lost_account_info = lost_acc or working_acc or {"uid": "unknown", "password": "unknown"}
                lost_accounts.append({**lost_account_info, "error_reason": error_msg})
                errors_summary[error_msg] += 1
                processed_count += 1
                continue

            processed_count += 1

            if token and working_acc:
                successful_tokens.append({"token": token, "region": region})
                working_accounts.append(working_acc)
            elif lost_acc:
                lost_accounts.append(lost_acc)
                reason = lost_acc.get("error_reason", "Unknown Failure")
                simple_error = reason.split(':')[0].strip()
                errors_summary[simple_error] += 1
            else:
                 logger.error(f"Task completed unexpectedly. Token:{token}, Region:{region}, Work:{working_acc}, Lost:{lost_acc}, Err:{error_reason}")
                 generic_lost_info = {"account_info": lost_acc or working_acc or "unknown", "error_reason": "Processing function returned unexpected state"}
                 lost_accounts.append(generic_lost_info)
                 errors_summary["Processing function error"] += 1

            # Progress Update
            current_time = time.time()
            update_frequency_items = max(10, min(100, total_count // 10))
            time_elapsed_since_last_update = current_time - last_update_time;

            if time_elapsed_since_last_update > 2.0 or \
               (update_frequency_items > 0 and processed_count % update_frequency_items == 0) or \
               processed_count == total_count:

                elapsed_time = current_time - start_time
                percentage = (processed_count / total_count) * 100 if total_count > 0 else 0

                estimated_remaining_time = -1
                if processed_count > 5 and elapsed_time > 2:
                    try:
                        time_per_item = elapsed_time / processed_count
                        remaining_items = total_count - processed_count
                        estimated_remaining_time = time_per_item * remaining_items
                    except ZeroDivisionError: pass

                progress_text = (
                    f"🔄 *Processing Accounts...*\n\n"
                    f"Progress: {processed_count}/{total_count} ({percentage:.1f}%)\n"
                    f"✅ Success: {len(successful_tokens)} | ❌ Failed: {len(lost_accounts)}\n"
                    f"⏱️ Elapsed: {format_time(elapsed_time)}\n"
                    f"⏳ Est. Remaining: {format_time(estimated_remaining_time)}"
                )

                if last_progress_text_sent != progress_text:
                    try:
                        await context.bot.edit_message_text(
                            chat_id=progress_message.chat_id, message_id=progress_message.message_id,
                            text=progress_text, parse_mode=ParseMode.MARKDOWN
                        )
                        last_progress_text_sent = progress_text
                        last_update_time = current_time
                    except TelegramError as edit_err:
                        if "Message is not modified" not in str(edit_err):
                             logger.warning(f"Could not edit progress message: {edit_err}")
                        last_update_time = current_time

    # Final Summary & File Generation
    final_elapsed_time = time.time() - start_time
    escaped_file_name = escape(file_name)
    final_summary_parts = [
        f"🏁 *Processing Complete for `{escaped_file_name}`*\n",
        f"📊 Total Accounts Processed: {total_count}",
        f"✅ Successful Tokens: {len(successful_tokens)}",
        f"❌ Failed/Invalid Accounts: {len(lost_accounts)}",
        f"⏱️ Total Time Taken: {format_time(final_elapsed_time)}"
    ]

    # Add Region Summary
    successful_by_region = defaultdict(list)
    if successful_tokens:
        for token_entry in successful_tokens:
            region = token_entry.get('region')
            region_name = region if region else "Unknown Region"
            successful_by_region[region_name].append(token_entry)

        if successful_by_region:
            final_summary_parts.append("\n*Successful by Region:*")
            sorted_regions = sorted(successful_by_region.keys())
            for region in sorted_regions:
                count = len(successful_by_region[region])
                final_summary_parts.append(f"- {escape(region)}: {count} tokens")
    else:
        final_summary_parts.append("\n*Successful by Region:* 0 tokens found.")

    # Add Error Summary
    if errors_summary:
        final_summary_parts.append("\n*Error Summary (Top 5 Types):*")
        sorted_errors = sorted(errors_summary.items(), key=lambda item: item[1], reverse=True)
        for msg, count in sorted_errors[:5]:
            final_summary_parts.append(f"- `{escape(msg)}`: {count} times")
        if len(sorted_errors) > 5:
            final_summary_parts.append(f"... and {len(sorted_errors) - 5} more error types.")

    final_summary = "\n".join(final_summary_parts)

    try:
        if progress_message:
            await context.bot.delete_message(chat_id=progress_message.chat_id, message_id=progress_message.message_id)
        await message.reply_text(
            final_summary,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_reply_markup
        )
    except TelegramError as final_msg_err:
        logger.error(f"Could not delete progress message or send final summary: {final_msg_err}")
        try:
            await message.reply_text(
                final_summary,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_reply_markup
            )
        except Exception as fallback_err:
            logger.critical(f"Failed even fallback sending final summary: {fallback_err}")

    # Generate and Send Output Files
    file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files_to_send = []
    cleanup_paths = []

    try:
        os.makedirs(TEMP_DIR, exist_ok=True)

        # Create main jwt_token.json file
        if successful_tokens:
            jwt_token_path = os.path.join(TEMP_DIR, f'jwt_only_{user_id}_{file_timestamp}.json')
            tokens_only_list_for_file = [{"token": entry.get("token")} for entry in successful_tokens if entry.get("token")]

            if tokens_only_list_for_file:
                if save_json_data(jwt_token_path, tokens_only_list_for_file):
                    output_files_to_send.append((jwt_token_path, 'jwt_token.json'))
                    cleanup_paths.append(jwt_token_path)
                else:
                    await message.reply_text("⚠️ Error saving main `jwt_token.json` to temporary storage.")

        # Create region-specific accounts{Region}.json files
        if successful_by_region:
            for region_name, entries in successful_by_region.items():
                 if not entries: continue

                 region_tokens_only = [{"token": entry.get("token")} for entry in entries if entry.get("token")]

                 if region_tokens_only:
                     sanitized_region_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', region_name).strip(' _.-')
                     if not sanitized_region_name.lower().endswith('.json'):
                         sanitized_region_name += ".json"
                     base_region_name = os.path.splitext(sanitized_region_name)[0]
                     region_file_name = f'accounts{base_region_name}.json'

                     region_file_path = os.path.join(TEMP_DIR, f'{base_region_name}_{user_id}_{file_timestamp}.json')

                     if save_json_data(region_file_path, region_tokens_only):
                         output_files_to_send.append((region_file_path, region_file_name))
                         cleanup_paths.append(region_file_path)
                     else:
                         await message.reply_text(f"⚠️ Error saving region file `{escape(region_file_name)}` to temporary storage.", parse_mode=ParseMode.MARKDOWN)

        # Create working_account.json
        if working_accounts:
            working_account_path = os.path.join(TEMP_DIR, f'working_{user_id}_{file_timestamp}.json')
            if save_json_data(working_account_path, working_accounts):
                output_files_to_send.append((working_account_path, 'working_account.json'))
                cleanup_paths.append(working_account_path)
            else:
                await message.reply_text("⚠️ Error saving `working_account.json` to temporary storage.")

        # Create lost_account.json
        if lost_accounts:
            lost_account_path = os.path.join(TEMP_DIR, f'lost_{user_id}_{file_timestamp}.json')
            if save_json_data(lost_account_path, lost_accounts):
                output_files_to_send.append((lost_account_path, 'lost_account.json'))
                cleanup_paths.append(lost_account_path)
            else:
                await message.reply_text("⚠️ Error saving `lost_account.json` to temporary storage.")

        # Send the generated files
        if output_files_to_send:
            await message.reply_text(f"⬇️ Sending {len(output_files_to_send)} result file(s)...")
            output_files_to_send.sort(key=lambda x: x[1])
            for temp_path, desired_filename in output_files_to_send:
                 if not os.path.exists(temp_path):
                     logger.error(f"Output file {temp_path} (for {desired_filename}) not found before sending.")
                     await message.reply_text(f"⚠️ Internal Error: Could not find `{escape(desired_filename)}` for sending.", parse_mode=ParseMode.MARKDOWN)
                     continue
                 try:
                     with open(temp_path, 'rb') as f:
                         await message.reply_document(
                             document=InputFile(f, filename=desired_filename),
                             caption=f"`{escape(desired_filename)}`\nFrom processing of: `{escaped_file_name}`\nTotal Processed: {total_count}",
                             parse_mode=ParseMode.MARKDOWN
                         )
                     logger.info(f"Sent '{desired_filename}' to user {user_id}")
                     await asyncio.sleep(0.5)
                 except TelegramError as send_err:
                     logger.error(f"Failed to send '{desired_filename}' to user {user_id}: {send_err}")
                     await message.reply_text(f"⚠️ Failed to send `{escape(desired_filename)}`: {escape(str(send_err))}", parse_mode=ParseMode.MARKDOWN)
                 except Exception as general_err:
                     logger.error(f"Unexpected error sending '{desired_filename}' to {user_id}: {general_err}")
                     await message.reply_text(f"⚠️ Unexpected error sending `{escape(desired_filename)}`.", parse_mode=ParseMode.MARKDOWN)
        elif total_count > 0:
             await message.reply_text("ℹ️ No output files were generated (e.g., 0 successful tokens found or error saving files).", reply_markup=main_reply_markup)

    except Exception as final_err:
        logger.error(f"Error during file generation/sending stage for user {user_id}: {final_err}")
        await message.reply_text(f"⚠️ An error occurred while generating/sending result files: {escape(str(final_err))}", reply_markup=main_reply_markup)
    finally:
        for path in cleanup_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.warning(f"Could not remove temp output file {path}: {e}")

# --- Global Error Handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)

    # Clean up potentially stuck user states on error
    if isinstance(update, Update) and update.effective_chat:
        chat_id_for_notify = update.effective_chat.id
        cleaned = False
        if context.user_data.pop('waiting_for_json', None):
             logger.info(f"Cleared 'waiting_for_json' state for chat {chat_id_for_notify} due to error.")
             cleaned = True

        if cleaned:
             try:
                 await context.bot.send_message(
                     chat_id=chat_id_for_notify,
                     text="⚠️ An internal error occurred. Any pending action has been cancelled. Please try again.",
                     reply_markup=main_reply_markup
                 )
             except Exception as notify_err:
                  logger.error(f"Failed to notify user {chat_id_for_notify} about state cleanup after error: {notify_err}")

# --- Main Application Setup ---

async def main() -> None:
    """Initialize data, set up handlers, and run the bot."""
    global TOKEN

    print("\n--- Initializing Bot ---")

    # Essential Config Checks
    if not TOKEN or TOKEN == "YOUR_FALLBACK_BOT_TOKEN":
        print("\n" + "="*60)
        print(" FATAL ERROR: TELEGRAM_BOT_TOKEN is missing or invalid.")
        print(" Please set the TELEGRAM_BOT_TOKEN environment variable or")
        print(" update the TOKEN variable directly in the script.")
        print(" -> Exiting.")
        print("="*60 + "\n")
        exit(1)
    elif len(TOKEN.split(':')) != 2:
        print("\n" + "="*60)
        print(f" FATAL ERROR: TELEGRAM_BOT_TOKEN format looks incorrect ('{TOKEN[:10]}...'). Should be 'ID:SECRET'.")
        print(" -> Exiting.")
        print("="*60 + "\n")
        exit(1)

    if not API_BASE_URL: logger.warning("JWT_API_URL not set, using default.")
    else: logger.info(f"Using API Base URL: {API_BASE_URL}")
    if not API_KEY or API_KEY == 'atxdev': logger.warning("JWT_API_KEY not set or using default.")

    # Create Directories
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
        logger.info(f"Data Directory: {DATA_DIR}")
        logger.info(f"Temp Directory: {TEMP_DIR}")
    except OSError as e:
        print(f"\nFATAL ERROR: Cannot create required directories: {e}\n-> Exiting.")
        exit(1)

    # Build Application
    app_builder = Application.builder().token(TOKEN)\
        .concurrent_updates(True) \
        .read_timeout(30) \
        .write_timeout(30) \
        .connect_timeout(30) \
        .pool_timeout(60) \
        .get_updates_read_timeout(40) \
        .get_updates_pool_timeout(70)

    application = app_builder.build()

    # Handlers
    private_chat_filter = filters.ChatType.PRIVATE

    # Core Commands & Buttons
    application.add_handler(CommandHandler("start", start, filters=private_chat_filter))
    application.add_handler(CommandHandler("help", help_command, filters=private_chat_filter))
    application.add_handler(MessageHandler(filters.Regex(f"^{re.escape(COMMAND_BUTTONS_LAYOUT[0][1])}$") & private_chat_filter, help_command))

    # Cancel Command & Button
    application.add_handler(CommandHandler("cancel", cancel, filters=private_chat_filter))
    application.add_handler(MessageHandler(filters.Regex(f"^{re.escape(COMMAND_BUTTONS_LAYOUT[1][0])}$") & private_chat_filter, cancel))

    # File Processing
    application.add_handler(MessageHandler(filters.Text(COMMAND_BUTTONS_LAYOUT[0][0]) & private_chat_filter, handle_document))
    application.add_handler(MessageHandler(
        (filters.Document.MimeType('application/json') | filters.Document.FileExtension('json')) & private_chat_filter,
        handle_document
    ))

    # Error Handler
    application.add_error_handler(error_handler)

    logger.info("🤖 Bot is initializing and connecting to Telegram...")
    print("\n" + "="*60)
    print(" 🚀 JWT Token Generator Bot is starting...")

    try:
        await application.initialize()

        bot_info = await application.bot.get_me()
        print(f" ✔️ Bot Username: @{bot_info.username} (ID: {bot_info.id})")
        print(f" ✔️ Developer: {DEVELOPER_USERNAME}")
        print(f" ✔️ Data Directory: {DATA_DIR}")

        await application.start()
        await application.updater.start_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
        print("\n Bot is now polling for updates. Press Ctrl+C to stop.")
        print("="*60 + "\n")

        # Keep the main thread alive
        while True:
            await asyncio.sleep(3600)

    except (TelegramError, ConnectionError) as e:
         print("\n" + "="*60)
         print(f" FATAL ERROR: Could not connect to Telegram or initialize bot.")
         print(f" Error: {e}")
         print(" Please check your network connection and bot token.")
         print(" -> Exiting.")
         print("="*60 + "\n")
         logger.critical(f"Failed to initialize or start polling: {e}")
         exit(1)
    except Exception as e:
        print("\n" + "="*60)
        print(f" FATAL ERROR: An unexpected error occurred during bot startup.")
        print(f" Error: {e}")
        print(" -> Exiting.")
        print("="*60 + "\n")
        logger.critical(f"Unhandled exception during startup: {e}")
        exit(1)
    finally:
         if 'application' in locals() and application.running:
              logger.info("Attempting graceful shutdown...")
              await application.stop()
              await application.shutdown()
              logger.info("Application stopped.")


if __name__ == '__main__':
    try:
        if not TOKEN or TOKEN == "YOUR_FALLBACK_BOT_TOKEN":
             print("FATAL: TELEGRAM_BOT_TOKEN is not set. Please configure it before running.")
        else:
             asyncio.run(main())
    except KeyboardInterrupt:
        print("\n-- Bot stopping due to Ctrl+C --")
        logger.info("Bot stopped manually via KeyboardInterrupt.")
    except Exception as e:
        print(f"\n💥 A critical unhandled exception occurred: {e}")
        logger.critical(f"Critical unhandled exception in __main__: {e}")