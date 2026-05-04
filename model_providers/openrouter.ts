/**
 * OpenRouter model provider for the ensemble system.
 */

import { OpenAIChat } from './openai_chat.js';
import OpenAI from 'openai';
import { appendJsonSchemaInstruction, getJsonSchemaFromResponseFormat } from '../utils/structured_output.js';

/**
 * OpenRouter model provider implementation
 */
export class OpenRouterProvider extends OpenAIChat {
    constructor() {
        super(
            'openrouter',
            process.env.OPENROUTER_API_KEY,
            'https://openrouter.ai/api/v1',
            {
                'User-Agent': 'JustEvery_',
                'HTTP-Referer': 'https://justevery.com/',
                'X-Title': 'JustEvery_',
            },
            {
                provider: {
                    require_parameters: true,
                    sort: 'throughput',
                    ignore: ['Novita'], // Fails frequently with Qwen tool calling
                },
            }
        );
    }

    prepareParameters(
        requestParams: OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming
    ): OpenAI.Chat.Completions.ChatCompletionCreateParamsStreaming {
        requestParams = super.prepareParameters(requestParams);

        const jsonSchema = getJsonSchemaFromResponseFormat(requestParams.response_format);
        if (jsonSchema && requestParams.model.startsWith('deepseek/')) {
            requestParams.response_format = { type: 'json_object' } as any;
            requestParams.messages = appendJsonSchemaInstruction(requestParams.messages, jsonSchema);
        }

        return requestParams;
    }
}

/**
 * A singleton instance of OpenRouterProvider for use in import statements
 */
export const openRouterProvider = new OpenRouterProvider();
