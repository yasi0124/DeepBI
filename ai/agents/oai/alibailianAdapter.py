# -*- coding: utf-8 -*-
"""
    阿里百炼平台适配，可以自行调整模型
    适配Openai 接口
    pip install dashscope

"""
try:
    import dashscope
except Exception as e:
    raise Exception("dashscope not install, please install dashscope,use commnd : pip install dashscope")
import copy
import json
import random
import time
from dashscope import Generation
from qwen_agent.llm import get_chat_model
from http import HTTPStatus

Ali_bailian_AI_MODEL = "qwen-max"

"""
 阿里role支持
    system 系统设定，必须在第一个消息，不能有多个，可以没有
    user 用户输入信息角色
    function 用户调用function_call(tools)结果内容
    assistant AI返回助手信息
    tool 这里是 ai返回需要调用的tool
"""


class AlibailianClient:

    @classmethod
    def run(cls, apiKey, data, model):
        if apiKey is None or apiKey == "":
            raise Exception("LLM DeepSeek apikey empty,use_model: ", model, " need apikey")
        dashscope.api_key = apiKey
        # call ai
        model = model if model is not None or "" == model else Ali_bailian_AI_MODEL
        result = cls.call_alibailian(data, model)
        return result


    @classmethod
    def run_local(cls, apiKey, data, model, apiHost):
        """
        基于qwen_agent请求个人部署的大模型服务
        """
        new_message = {}
        messages_copy = copy.deepcopy(data['messages'])
        # post请求体格式适配，openai格式 ->qwen格式
        new_message['messages'] = cls.input_dashscope_to_qwenagent(messages_copy)
        llm = get_chat_model({
            'model': model,
            'model_server': 'http://36.140.46.179:8009/v1',
            'api_key': apiKey,
        })
        responses = []
        try:
            if "functions" in data:
                for responses in llm.chat(messages=new_message['messages'], functions=data['functions'], stream=True):
                    pass
            else:
                for responses in llm.chat(messages=new_message['messages'], stream=True):
                    pass
            # 返回格式适配 qwen格式 -> dashscope格式
            response = cls.output_qwentagent_to_dashscope(responses[-1])
            response = cls.output_to_openai(response, model)
            return response
        except Exception as e:
            raise Exception(str(e))

    @classmethod
    def call_alibailian(cls, data, model):
        new_message = {}
        messages_copy = copy.deepcopy(data['messages'])
        new_message['messages'] = cls.input_to_openai(messages_copy)

        try:
            if "functions" in data:
                response = Generation.call(model=model,
                                           tools=data['functions'],
                                           messages=new_message['messages'],
                                           seed=random.randint(1, 10000),
                                           result_format='message')
            else:
                response = Generation.call(model=model,
                                           messages=new_message['messages'],
                                           seed=random.randint(1, 10000),
                                           result_format='message')
            if response.status_code == HTTPStatus.OK:
                return cls.output_to_openai(response, model)
            else:
                raise Exception('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                    response.request_id, response.status_code,
                    response.code, response.message
                ))
        except Exception as e:
            raise Exception(str(e))


    @classmethod
    def input_to_openai(cls, messages):
        """
            将输入的openai message转换为现在模型的message格式,这里主要处理的是function call 和 role
        """
        # change role and function name
        transformed_message = []
        for message in messages:
            # update role system to user
            if message['role'] == "system":
                message['role'] = "user"
            elif message['role'] == "function":
                # update function_call result role
                message['role'] = "tool"
            # update function_call to tool message
            if "function_call" in message:
                message.pop("name")
                tool_calls = [
                    {
                        "function": message.pop('function_call'),
                        "id": "",
                        "type": "function"
                    }
                ]
                message['tool_calls'] = tool_calls
            transformed_message.append(message)
        result = []
        # marge message
        now_role = ""
        now_content = ""
        result = []
        for i, item in enumerate(transformed_message):
            content = item.get('content')
            role = item.get("role")

            # other not first system
            if now_role != "":
                if role == now_role and role != "tool":
                    now_content = now_content + "\n" + content
                else:
                    now_item = item
                    now_item['content'] = now_content
                    now_item['role'] = now_role
                    # add old item
                    result.append(now_item)
                    # process new item
                    now_role = role
                    now_content = content
            else:
                now_role = role
                now_content = content
            if i == len(transformed_message) - 1:
                now_item = item
                now_item['content'] = now_content
                now_item['role'] = now_role
                result.append(now_item)
        return result
        pass

    @classmethod
    def output_to_openai(cls, data, model):
        """
            将输出的message转换为openai的message格式,这里主要处理的是function call 和 role
        """
        result = {}
        result['status_code'] = "200"
        result['id'] = data["request_id"]
        result['model'] = model
        result['created'] = int(time.time())
        result['object'] = "chat.completion"
        output = data["output"]
        if output['choices'][0]['finish_reason'] == "tool_calls":
            # function call
            choices = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": output['choices'][0]['message']['tool_calls'][0]['function']['name'],
                        "arguments": output['choices'][0]['message']['tool_calls'][0]['function']['arguments']
                    }
                },
                "finish_reason": "function_call"  # 原来是tools_call
            }

        else:
            # message call
            choices = {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output['choices'][0]['message']['content'],
                },
                "finish_reason": "stop"  # 原来是tools_call
            }
            pass
        result['choices'] = [choices]
        result['usage'] = {
            "prompt_tokens": data['usage']['input_tokens'],
            "completion_tokens": data['usage']['output_tokens'],
            "total_tokens": data['usage']['input_tokens'] + data['usage']['output_tokens']
        }
        return result

    @classmethod
    def input_dashscope_to_qwenagent(cls, messages):
        """根据大模型服务的数据校验进行适配"""
        new_message = []
        for message in messages:
            # update role system to user role角色全部换成user，如果role是system有额外校验
            if message['role'] == "system":
                message['role'] = "user"
            tmp_mess = {
                "role": message.get('role', ""),
                "name": message.get('name', ""),
                "content": message.get('content', "")
            }
            # function_call 如果存在不能为空
            if "function_call" in messages and message['function_call']:
                tmp_mess["function_call"] = message['function_call']
            new_message.append(tmp_mess)
        return new_message

    @classmethod
    def output_qwentagent_to_dashscope(cls, response: dict = None):
        # 返回体从qwen格式 -> dashscope格式。下面的模板是默认的dashcope格式
        function_call = response.get("function_call", None)
        # not yet为当前我们返回体中没有相应字段适配
        dashscope_template = {
            "status_code": 200,
            "request_id": "not yet",
            "code": "",
            "message": "",
            "output": {
                "text": None,
                "finish_reason": None,
                "choices": [
                    {
                        "finish_reason": "tool_calls" if "function_call" in response else "",
                        "message": {
                            "role": response.get("role", "default"),
                            "content": response.get("content", "default"),
                            "tool_calls": [
                            {
                                "function": {
                                    "name": response.get("function_call", {}).get("name", ""),
                                    "arguments": response.get("function_call", {}).get("arguments", "")
                                },
                                "id": "",
                                "type": "function"
                            }]
                        }
                    }]
            },
            "usage": {
                "input_tokens": "not yet",
                "output_tokens": "not yet",
                "total_tokens": "not yet"
            }
        }
        if function_call is None:
            dashscope_template["output"]["choices"][0]["message"].pop("tool_calls")
        return dashscope_template
