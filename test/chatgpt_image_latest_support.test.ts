import { beforeEach, describe, expect, it, vi } from 'vitest';
import { findModel } from '../data/model_data.js';
import { OpenAIProvider } from '../model_providers/openai.js';
import { costTracker } from '../utils/cost_tracker.js';

describe('chatgpt-image-latest support', () => {
    beforeEach(() => {
        costTracker.reset();
    });

    it('registers chatgpt-image-latest with OpenAI image pricing metadata', () => {
        expect(findModel('chatgpt-image-latest')).toMatchObject({
            id: 'chatgpt-image-latest',
            provider: 'openai',
            cost: {
                per_image: 0.034,
                input_per_million: {
                    text: 5.0,
                    image: 8.0,
                },
                output_per_million: {
                    text: 10.0,
                    image: 32.0,
                },
            },
        });
    });

    it('passes chatgpt-image-latest directly to OpenAI image generation and tracks published pricing', async () => {
        const provider = new OpenAIProvider('sk-test');
        const generate = vi.fn().mockResolvedValue({
            data: [{ b64_json: 'YWJjMTIz' }],
        });

        (provider as any)._client = {
            images: {
                generate,
            },
        };

        const images = await provider.createImage(
            'A bold editorial poster with sharp geometry',
            'chatgpt-image-latest',
            { agent_id: 'test-chatgpt-image-latest' } as any,
            {
                quality: 'medium',
                size: '1536x1024',
            }
        );

        expect(images).toEqual(['data:image/png;base64,YWJjMTIz']);
        expect(generate).toHaveBeenCalledWith({
            model: 'chatgpt-image-latest',
            prompt: 'A bold editorial poster with sharp geometry',
            n: 1,
            background: 'auto',
            quality: 'medium',
            size: '1536x1024',
            moderation: 'low',
            output_format: 'png',
        });

        const costsByModel = costTracker.getCostsByModel();
        expect(costsByModel['chatgpt-image-latest']?.cost).toBeCloseTo(0.05);
        expect(costsByModel['chatgpt-image-latest']?.calls).toBe(1);
    });
});
