import requests as requests

api_key = 'sk-GWWHEy4N6FE0F37E3EcbT3BlBkFJc7a826C2d307420d8733'  # 申请的API Key
llm_model = 'glm-4-flash'  # 想用的llm

# def get_translate(query, from_l, to_l, api_key=api_key, llm_model=llm_model):
#     """
#     Translate a query from one language to another using the specified LLM API.

#     Args:
#         query (str): The text to be translated.
#         from_l (str): The source language.
#         to_l (str): The target language.
#         api_key (str): The API key for authentication.
#         llm_model (str): The language model to use.

#     Returns:
#         str: The translated text.
#     """
#     headers = {
#         "Authorization": 'Bearer ' + api_key,
#     }

#     # Construct the prompt for the translation task
#     prompt = f"Translate the following text from {from_l} to {to_l}. Provide only one accurate and faithful translation. Do not include explanations, comments, additional content or multiple versions:\n\n{query}"

#     # Set up the request parameters
#     params = {
#         "messages": [
#             {
#                 "role": 'user',
#                 "content": prompt
#             }
#         ],
#         "model": llm_model
#     }

#     # Make the API request
#     response = requests.post(
#         "https://aigptx.top/v1/chat/completions",
#         headers=headers,
#         json=params,
#         stream=False
#     )

#     # Parse the response content
#     res = response.json()
#     if 'choices' in res and len(res['choices']) > 0:
#         return res['choices'][0]['message']['content'].strip()
#     else:
#         raise ValueError("Failed to get a valid response from the API.")

def chat(system_prompts,user_prompt):
    api_key = 'sk-GWWHEy4N6FE0F37E3EcbT3BlBkFJc7a826C2d307420d8733'
    headers = {
        "Authorization": 'Bearer ' + api_key,
    }

    params = {
        "messages": [
            {
                "role": 'system',
                "content": system_prompts
            },
            {
                "role": 'user',
                "content": user_prompt
            }
        ],
        # 如果需要切换模型，在这里修改
        "model": 'glm-4-flash'
    }
    response = requests.post(
    "https://aigptx.top/v1/chat/completions",
    headers=headers,
    json=params,
    stream=False
    )
    res = response.json()
    res_content = res['choices'][0]['message']['content']
    # print(res_content)
    return res_content

if __name__ == '__main__':
    # import requests as requests

    # # 在这里配置您在本站的API_KEY
    # api_key = api_key

    # headers = {
    #     "Authorization": 'Bearer ' + api_key,
    # }

    # question = "1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1=？\n"

    # params = {
    #     "messages": [

    #         {
    #             "role": 'user',
    #             "content": question
    #         }
    #     ],
    #     # 如果需要切换模型，在这里修改
    #     "model": llm_model
    # }
    # response = requests.post(
    #     "https://aigptx.top/v1/chat/completions",
    #     headers=headers,
    #     json=params,
    #     stream=False
    # )
    # res = response.json()
    # res_content = res['choices'][0]['message']['content']
    # print(res_content)
    res = chat("你是一个温柔体贴的女大学生","你是谁？")
    print(res)
