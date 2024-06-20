#按照目标语言构建提示模版

JA_PROMPT = {
    "prompt": "###  次の{}のテキストを日本語に翻訳してください：\n{}：\n",
    "response": "\n###  日本語：\n######\n",
    "JA": "日本語",
    "EN": "英語",
    "ZH": "中国語",
    "FR": "フランス語",
    "DE": "ドイツ語",
    "ES": "スペイン語",
    "IT": "イタリア語",
    "PT": "ポルトガル語",
    "RU": "ロシア語",
    "NL": "オランダ語",
    "PL": "ポーランド語",
    "TR": "トルコ語",
    "AR": "アラビア語",
    "VI": "ベトナム語",
    "TH": "タイ語",
    "ID": "インドネシア語",
    "MS": "マレー語",
}


EN_PROMPT = {
    "prompt": "###  Please translate the following {} text into English:\n{}:\n",
    "response": "\n###  English:\n######\n",
    "JA": "Japanese",
    "ZH": "Chinese",
    "FR": "French",
    "DE": "German",
    "ES": "Spanish",
    "IT": "Italian",
    "PT": "Portuguese",
    "RU": "Russian",
    "NL": "Dutch",
    "PL": "Polish",
    "TR": "Turkish",
    "AR": "Arabic",
    "VI": "Vietnamese",
    "TH": "Thai",
    "ID": "Indonesian",
    "MS": "Malay",
}

ZH_PROMPT = {
    "prompt": "###  请将以下的{}文本翻译成中文：\n{}：\n",
    "response": "\n###  中文：\n######\n",
    "EN": "英语",
    "JA": "日语",
    "FR": "法语",
    "DE": "德语",
    "ES": "西班牙语",
    "IT": "意大利语",
    "PT": "葡萄牙语",
    "RU": "俄语",
    "NL": "荷兰语",
    "PL": "波兰语",
    "TR": "土耳其语",
    "AR": "阿拉伯语",
    "VI": "越南语",
    "TH": "泰语",
    "ID": "印尼语",
    "MS": "马来语",
}

lang_code2prompt = {
    "JA": JA_PROMPT,
    "EN": EN_PROMPT,
    "ZH": ZH_PROMPT,
}

end_of_prompt = "<|reserved_special_token_249|>"


def get_prefix_response_template(src_lang, tgt_lang):
    tgt_prompt = lang_code2prompt[tgt_lang]
    prefix = tgt_prompt["prompt"].format(tgt_prompt[src_lang], tgt_prompt[src_lang])
    response_template = tgt_prompt["response"]
    return prefix, response_template

def formatting_prompts_func(examples):
    output_texts=[]
    for src, tgt,src_lang, tgt_lang in zip(examples["src"], examples["tgt"], examples["src_lang"], examples["tgt_lang"]):
        prefix, response_template = get_prefix_response_template(src_lang, tgt_lang)
        text = f"{prefix}{src}{response_template}{tgt}<|end_of_text|>"
        output_texts.append(text)
    outputs= {"text": output_texts}
    return outputs


def formatting_prompts_func_eval(examples):
    src, tgt,src_lang, tgt_lang = (examples["src"], examples["tgt"], examples["src_lang"], examples["tgt_lang"])
    prefix, response_template = get_prefix_response_template(src_lang, tgt_lang)
    text = f"{prefix}{src}{response_template}"
    return textfix}{src}{response_template}{end_of_prompt}"
    return text
