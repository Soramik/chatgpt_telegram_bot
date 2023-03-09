import config

import openai
openai.api_key = config.openai_api_key


CHAT_MODES = {
    "助理": {
        "name": "👩🏼‍🎓 ChatGPT 助理",
        "welcome_message": "👩🏼‍🎓 你好, 我是 <b>ChatGPT 助理</b> 。 请问您需要什么帮助？",
        "prompt_start": "作为一款名为ChatGPT的高级聊天机器人，你的主要目标是在你所能做到的最好的范围内协助用户。这可能涉及回答问题、提供有用信息或根据用户输入完成任务。为了有效地协助用户，重要的是要在回复中详细和全面地表达自己的意见。使用例子和证据来支持你的观点，并为你的建议或解决方案提供理由。记得始终把用户的需求和满意度放在首位。你的最终目标是为用户提供一个有益和愉悦的体验。"
    },

    "代码助理": {
        "name": "👩🏼‍💻 代码助理",
        "welcome_message": "👩🏼‍💻 你好，我是 <b>ChatGPT 代码助理</b> 。 请问您需要什么帮助？",
        "prompt_start": "作为一款名为ChatGPT的高级聊天机器人，你的主要目标是协助用户编写代码。这可能涉及到设计、编写、编辑、描述代码或提供有用信息。如果可能的话，你应该提供代码示例来支持你的观点并为你的建议或解决方案提供理由。确保你提供的代码是正确的，并且可以无错误地运行。在回复中详细和全面地表达自己的意见。你的最终目标是为用户提供一个有益和愉悦的体验。请将代码写在<code></code>标签中。"
    },

    "文本改进助理": {
        "name": "📝 文本改进助理",
        "welcome_message": "📝 你好，我是 <b>ChatGPT 文本改进助理</b> 。 您可以将任何文本发送给我，我会尽可能改进它并纠正所有的错误。",
        "prompt_start": "作为一款名为ChatGPT的高级聊天机器人，你的主要目标是纠正用户发送的拼写错误、修复错误并改进文本。你的目标是编辑文本，而不是改变它的意思。你可以用更漂亮、更优美、更高级的词汇和句子替换简化的A0级别的单词和句子。你的所有答案严格遵循以下结构（保留HTML标记）：\n<b>编辑后的文本：</b>\n{编辑后的文本}\n\n<b>更正：</b>\n{有关文本更正的编号列表}"
    },
}

OPENAI_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0
}


class ChatGPT:
    def __init__(self, use_chatgpt_api=True):
        self.use_chatgpt_api = use_chatgpt_api
    
    async def send_message(self, message, dialog_messages=[], chat_mode="助理"):
        if chat_mode not in CHAT_MODES.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.use_chatgpt_api:
                    messages = self._generate_prompt_messages_for_chatgpt_api(message, dialog_messages, chat_mode)
                    r = await openai.ChatCompletion.acreate(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].message["content"]
                else:
                    prompt = self._generate_prompt(message, dialog_messages, chat_mode)
                    r = await openai.Completion.acreate(
                        engine="text-davinci-003",
                        prompt=prompt,
                        **OPENAI_COMPLETION_OPTIONS
                    )
                    answer = r.choices[0].text

                answer = self._postprocess_answer(answer)
                n_used_tokens = r.usage.total_tokens
                
            except openai.error.InvalidRequestError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e

                # forget first message in dialog_messages
                dialog_messages = dialog_messages[1:]

        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, n_used_tokens, n_first_dialog_messages_removed

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = CHAT_MODES[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"ChatGPT: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "ChatGPT: "

        return prompt

    def _generate_prompt_messages_for_chatgpt_api(self, message, dialog_messages, chat_mode):
        prompt = CHAT_MODES[chat_mode]["prompt_start"]
        
        messages = [{"role": "system", "content": prompt}]
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer


async def transcribe_audio(audio_file):
    r = await openai.Audio.atranscribe("whisper-1", audio_file)
    return r["text"]