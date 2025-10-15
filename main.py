import telebot
import pandas as pd
import os
import tempfile
import google.generativeai as genai
from io import StringIO
import logging
import re
import json
import subprocess  # For allowing pip installs in generated code
from dotenv import load_dotenv

# Load environment variables from .env file (optional, kept for compatibility)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='bot.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Hard-coded Telegram Bot Token (replace 'YOUR_TELEGRAM_BOT_TOKEN_HERE' with your actual token)
TELEGRAM_BOT_TOKEN = '8218669109:AAF6g0N-31xjhSmBtSG0vUjhg-wvMpG3Cj0'

# Hard-coded Gemini API key (replace 'YOUR_GEMINI_API_KEY_HERE' with your actual API key)
GEMINI_API_KEY = 'AIzaSyC-D2P1rJlRUv-q_RwY-g-Zrwbz2hHNm-k'

# Validate that API keys are set
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

# Configure the Gemini API with the hard-coded key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')  # Using powerful Gemini 2.5 Pro model

# Define JSON schema for raw code output
CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"}
    },
    "required": ["code"]
}

# Define JSON schema for plan output
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"}
    },
    "required": ["plan"]
}

# For LLM integration using Gemini API with structured JSON output and fallback extraction
def call_llm(prompt, schema=CODE_SCHEMA):
    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        response_text = response.text.strip()
        logging.debug(f"Raw response text: {response_text}")

        # Try to parse as JSON
        try:
            response_json = json.loads(response_text)
            key = "code" if schema == CODE_SCHEMA else "plan"
            result = response_json.get(key, "")
            if result:
                logging.debug(f"Extracted {key} from JSON: {result}")
                return result.strip()
        except json.JSONDecodeError:
            logging.warning("Response not valid JSON; falling back to extraction.")

        # Fallback: Extract using regex (for code or plan)
        pattern = r'(def process_file\(file_path\):.*?return.*?)' if schema == CODE_SCHEMA else r'(.*)'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            result = match.group(1).strip() if schema == CODE_SCHEMA else match.group(0).strip()
            logging.debug(f"Extracted with regex: {result}")
            return result

        # Last resort: Strip wrappers
        result = re.sub(r'^\s*(\.\.\.|ellipsis|etc\.)\s*$', '', response_text, flags=re.IGNORECASE)
        result = re.sub(r'```(?:python|json)?\s*\n?(.*?)\n?```', r'\1', result, flags=re.DOTALL | re.IGNORECASE)
        result = result.strip()
        if ('def process_file' in result and schema == CODE_SCHEMA) or (schema == PLAN_SCHEMA):
            logging.debug(f"Last resort extracted: {result}")
            return result
        else:
            raise ValueError("No extractable content found in response")

    except Exception as e:
        logging.error(f"Error calling Gemini API: {str(e)}")
        raise Exception(f"Error calling Gemini API: {str(e)}")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Dictionary to store user states (e.g., waiting for description after file upload)
user_states = {}

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Welcome! Upload a file (TXT, CSV, XLSX) and then describe what you want to do with it (e.g., 'Extract the first column of numbers and split into TXT files with 100 lines each').")

@bot.message_handler(content_types=['document'])
def handle_document(message):
    try:
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Save to temp file
        _, file_extension = os.path.splitext(message.document.file_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(downloaded_file)
            file_path = temp_file.name

        # Log file info
        logging.debug(f"Uploaded file: {message.document.file_name}, size: {message.document.file_size}, path: {file_path}")

        # Store the file path and set state to wait for description
        user_states[message.chat.id] = {'file_path': file_path, 'state': 'waiting_for_description'}

        bot.reply_to(message, f"File '{message.document.file_name}' uploaded. Now, describe what you want to do with it (e.g., 'Extract the first column of numbers and split into TXT files with 100 lines each').")
    except Exception as e:
        logging.error(f"Error handling file: {str(e)}")
        bot.reply_to(message, f"Error handling file: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_description(message):
    chat_id = message.chat.id
    if chat_id in user_states and user_states[chat_id].get('state') == 'waiting_for_description':
        file_path = user_states[chat_id]['file_path']
        description = message.text.lower().strip()  # Normalize description for matching
        logging.debug(f"User description: {description}")

        # Detect if complex task (e.g., long description or contains 'complex')
        is_complex = len(description) > 50 or 'complex' in description

        # Fallback for simple "convert/change to txt" requests
        if any(keyword in description for keyword in ['convert', 'change']) and 'txt' in description:
            try:
                def process_file_fallback(file_path):
                    try:
                        if file_path.endswith('.csv'):
                            df = pd.read_csv(file_path, dtype=str)
                        elif file_path.endswith('.xlsx'):
                            df = pd.read_excel(file_path, dtype=str)
                        else:
                            raise ValueError("Unsupported file type")

                        # Log dataframe info for debugging
                        logging.debug(f"Dataframe shape: {df.shape}, first column: {df.iloc[:, 0].head().tolist()}")

                        # Extract first column, drop NaNs, convert to str
                        numbers = df.iloc[:, 0].dropna().astype(str).tolist()
                        if not numbers:
                            raise ValueError("First column is empty or contains no valid data")

                        # Split into chunks of 100
                        chunks = [numbers[i:i+100] for i in range(0, len(numbers), 100)]
                        output_files = []

                        for idx, chunk in enumerate(chunks):
                            output_path = f'chunk_{idx+1}.txt'
                            with open(output_path, 'w') as f:
                                f.write('\n'.join(chunk) + '\n')  # Add trailing newline
                            output_files.append(output_path)

                        logging.debug(f"Generated {len(output_files)} output files: {output_files}")
                        return output_files
                    except Exception as e:
                        logging.error(f"Fallback processing error: {str(e)}")
                        raise Exception(f"Error in fallback processing: {str(e)}")

                output_files = process_file_fallback(file_path)

                # Send output files back to user
                for out_file in output_files:
                    if os.path.exists(out_file):
                        with open(out_file, 'rb') as f:
                            bot.send_document(chat_id, f, caption=os.path.basename(out_file))
                        os.remove(out_file)  # Clean up
                    else:
                        logging.warning(f"Output file not found: {out_file}")

                bot.reply_to(message, f"Processing complete using fallback for '{description}'. Here are the TXT chunk files (100 numbers each, one per line).")

                # Clean up temp file and state
                os.remove(file_path)
                del user_states[chat_id]
                return
            except Exception as e:
                bot.reply_to(message, f"Fallback error: {str(e)}")
                logging.error(f"Fallback error: {str(e)}")

        # For general/complex cases, use LLM
        plan = ""
        if is_complex:
            # Generate plan first
            plan_prompt = f"""
The user uploaded a file at '{file_path}' (could be TXT, CSV, or XLSX).
The user's request: '{description}'.

Create a step-by-step plan for processing the file in Python. Break down the task into logical steps, including any libraries needed (install via subprocess if not available), error handling, and output file generation.
Output as JSON: {{"plan": "the detailed plan as a string"}}
Strictly JSON only.
"""
            try:
                plan = call_llm(plan_prompt, schema=PLAN_SCHEMA)
                logging.debug(f"Generated plan: {plan}")
            except Exception as e:
                bot.reply_to(message, f"Failed to generate plan: {str(e)}")
                logging.error(f"Failed to generate plan: {str(e)}")
                os.remove(file_path)
                del user_states[chat_id]
                return

        # Create prompt for Gemini to generate Python code, including plan if available
        code_prompt = f"""
The user uploaded a file at '{file_path}' (could be TXT, CSV, or XLSX).
The user's request: '{description}'.
{'Plan: ' + plan if plan else ''}

Generate a Python function named 'process_file' that takes 'file_path' as input, processes the file accordingly, and returns a list of output file paths (e.g., generated TXT files).
Use libraries like pandas if needed (assume imported as pd). If additional libraries are required, use import subprocess; subprocess.call(['pip', 'install', 'lib']) at the start.
Handle file types appropriately.
For example, if extracting first column numbers and splitting into chunks of 100:
- Read the file with pd.read_csv or pd.read_excel.
- Extract df.iloc[:, 0], convert to list.
- Split into chunks.
- Write each chunk to a separate TXT file with one item per line.
- Return the list of TXT file paths.

Ensure the code is safe, doesn't delete input files, and handles errors.
Output as JSON: {{"code": "the exact Python code for the function, no imports outside the function, start with def process_file(file_path):"}}
Do not include any explanations, markdown, or extra text. Strictly JSON only.
"""

        max_attempts = 3
        attempt = 0
        generated_code = None
        while attempt < max_attempts:
            try:
                generated_code = call_llm(code_prompt)
                break
            except Exception as e:
                bot.reply_to(message, f"Failed to generate code on attempt {attempt + 1}: {str(e)}")
                logging.error(f"Failed to generate code on attempt {attempt + 1}: {str(e)}")
                attempt += 1
                if attempt == max_attempts:
                    bot.reply_to(message, "Max attempts reached. Unable to generate valid code.")
                    os.remove(file_path)
                    del user_states[chat_id]
                    return

        last_error = None
        for attempt in range(max_attempts):
            try:
                # Execute the generated code safely
                local_namespace = {'pd': pd, 'os': os, 'subprocess': subprocess}
                exec(generated_code, {}, local_namespace)
                process_file = local_namespace.get('process_file')

                if callable(process_file):
                    output_files = process_file(file_path)

                    if output_files:
                        # Send output files back to user
                        for out_file in output_files:
                            if os.path.exists(out_file):
                                with open(out_file, 'rb') as f:
                                    bot.send_document(chat_id, f, caption=os.path.basename(out_file))
                                os.remove(out_file)  # Clean up
                            else:
                                logging.warning(f"Output file not found: {out_file}")

                    bot.reply_to(message, "Processing complete! Here are the output files.")
                    break
                else:
                    raise ValueError("Generated code did not define 'process_file' function")
            except SyntaxError as se:
                last_error = f"Syntax error (line {se.lineno}): {str(se)}"
            except Exception as e:
                last_error = str(e)

            if last_error and attempt < max_attempts - 1:
                # Feedback loop: Send error back to Gemini to fix
                fix_prompt = f"""
Previous code: {generated_code}
Error: {last_error}

Fix the code based on the error. Keep the same structure and requirements.
Output as JSON: {{"code": "the fixed Python code"}}
Strictly JSON only.
"""
                try:
                    generated_code = call_llm(fix_prompt)
                    logging.debug(f"Fixed code on attempt {attempt + 1}: {generated_code}")
                except Exception as fix_e:
                    bot.reply_to(message, f"Failed to fix code: {str(fix_e)}")
                    logging.error(f"Failed to fix code: {str(fix_e)}")
                    break
            else:
                bot.reply_to(message, f"Error executing script after {attempt + 1} attempts: {last_error}\n\nGenerated code for debugging:\n```{generated_code}```")
                logging.error(f"Final error after attempts: {last_error}\nGenerated code: {generated_code}")
                break

        # Clean up temp file and state
        try:
            os.remove(file_path)
        except Exception as e:
            logging.error(f"Error removing temp file: {str(e)}")
        if chat_id in user_states:
            del user_states[chat_id]
    else:
        bot.reply_to(message, "Please upload a file first using the document attachment.")

# Start the bot
if __name__ == "__main__":
    bot.polling()
