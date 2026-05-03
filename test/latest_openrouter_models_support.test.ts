import { describe, expect, it } from 'vitest';
import { findModel } from '../data/model_data.js';
import { getModelFromAgent, getProviderFromModel } from '../model_providers/model_provider.js';

describe('latest OpenRouter model support', () => {
    it('registers GLM-5.1 with OpenRouter pricing and aliases', async () => {
        const model = findModel('GLM-5.1');

        expect(model?.id).toBe('z-ai/glm-5.1');
        expect(await getModelFromAgent({ agent_id: 'glm', model: 'glm-5.1' } as any)).toBe('z-ai/glm-5.1');
        expect(getProviderFromModel('z-ai/glm-5.1')).toBe('openrouter');
        expect(model?.cost).toMatchObject({
            input_per_million: 1.05,
            cached_input_per_million: 0.525,
            output_per_million: 3.5,
        });
        expect(model?.features).toMatchObject({
            context_length: 202752,
            max_output_tokens: 65535,
            input_modality: ['text'],
            output_modality: ['text'],
            tool_use: true,
            streaming: true,
            json_output: true,
            reasoning_output: true,
        });
    });

    it('registers Kimi K2.6 with multimodal OpenRouter details', async () => {
        const model = findModel('Kimi K2.6');

        expect(model?.id).toBe('moonshotai/kimi-k2.6');
        expect(await getModelFromAgent({ agent_id: 'kimi', model: 'kimi-k2-6' } as any)).toBe('moonshotai/kimi-k2.6');
        expect(getProviderFromModel('moonshotai/kimi-k2.6')).toBe('openrouter');
        expect(model?.cost).toMatchObject({
            input_per_million: 0.74,
            cached_input_per_million: 0.14,
            output_per_million: 3.49,
        });
        expect(model?.features).toMatchObject({
            context_length: 262142,
            max_output_tokens: 262142,
            input_modality: ['text', 'image'],
            output_modality: ['text'],
            tool_use: true,
            streaming: true,
            json_output: true,
            reasoning_output: true,
        });
    });

    it('registers DeepSeek V4 Pro and Flash as OpenRouter models', async () => {
        const pro = findModel('DeepSeek-V4');
        const flash = findModel('DeepSeek-V4-Flash');

        expect(pro?.id).toBe('deepseek/deepseek-v4-pro');
        expect(await getModelFromAgent({ agent_id: 'deepseek-pro', model: 'deepseek-v4-pro' } as any)).toBe(
            'deepseek/deepseek-v4-pro'
        );
        expect(getProviderFromModel('deepseek/deepseek-v4-pro')).toBe('openrouter');
        expect(pro?.cost).toMatchObject({
            input_per_million: 0.435,
            cached_input_per_million: 0.003625,
            output_per_million: 0.87,
        });
        expect(pro?.features).toMatchObject({
            context_length: 1048576,
            max_output_tokens: 384000,
            input_modality: ['text'],
            output_modality: ['text'],
            tool_use: true,
            streaming: true,
            json_output: true,
            reasoning_output: true,
        });

        expect(flash?.id).toBe('deepseek/deepseek-v4-flash');
        expect(await getModelFromAgent({ agent_id: 'deepseek-flash', model: 'deepseek-v4-flash' } as any)).toBe(
            'deepseek/deepseek-v4-flash'
        );
        expect(flash?.cost).toMatchObject({
            input_per_million: 0.14,
            cached_input_per_million: 0.0028,
            output_per_million: 0.28,
        });
    });

    it('registers Qwen 3.6 Plus and current smaller Qwen 3.6 variants', async () => {
        const plus = findModel('Qwen 3.6');
        const a3b = findModel('qwen3.6-35b-a3b');
        const dense = findModel('qwen3.6-27b');

        expect(plus?.id).toBe('qwen/qwen3.6-plus');
        expect(await getModelFromAgent({ agent_id: 'qwen-plus', model: 'qwen-3.6-plus' } as any)).toBe(
            'qwen/qwen3.6-plus'
        );
        expect(getProviderFromModel('qwen/qwen3.6-plus')).toBe('openrouter');
        expect(plus?.cost).toMatchObject({
            input_per_million: 0.325,
            output_per_million: 1.95,
        });
        expect(plus?.features).toMatchObject({
            context_length: 1000000,
            max_output_tokens: 65536,
            input_modality: ['text', 'image', 'video'],
            output_modality: ['text'],
            tool_use: true,
            streaming: true,
            json_output: true,
            reasoning_output: true,
        });

        expect(a3b?.id).toBe('qwen/qwen3.6-35b-a3b');
        expect(a3b?.cost).toMatchObject({
            input_per_million: 0.1612,
            cached_input_per_million: 0.1612,
            output_per_million: 0.96525,
        });
        expect(a3b?.features?.max_output_tokens).toBe(65536);

        expect(dense?.id).toBe('qwen/qwen3.6-27b');
        expect(dense?.cost).toMatchObject({
            input_per_million: 0.32,
            output_per_million: 3.2,
        });
        expect(dense?.features?.max_output_tokens).toBe(81920);
    });
});
