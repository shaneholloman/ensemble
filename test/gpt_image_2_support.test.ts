import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { findModel, MODEL_CLASSES } from '../data/model_data.js';
import { OpenAIProvider } from '../model_providers/openai.js';
import { costTracker } from '../utils/cost_tracker.js';

describe('gpt-image-2 support', () => {
    beforeEach(() => {
        costTracker.reset();
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('registers gpt-image-2 with OpenAI token pricing and aliases', () => {
        expect(MODEL_CLASSES.image_generation.models[0]).toBe('gpt-image-2');
        expect(findModel('gpt-image-2')).toMatchObject({
            id: 'gpt-image-2',
            aliases: ['gpt-image-2-2026-04-21'],
            provider: 'openai',
            cost: {
                per_image: 0.053,
                input_per_million: {
                    text: 5.0,
                    image: 8.0,
                },
                cached_input_per_million: {
                    text: 1.25,
                    image: 2.0,
                },
                output_per_million: {
                    text: 10.0,
                    image: 30.0,
                },
            },
            features: {
                input_modality: ['text', 'image'],
                output_modality: ['image'],
            },
            class: 'image_generation',
        });
        expect(findModel('gpt-image-2-2026-04-21')?.id).toBe('gpt-image-2');
    });

    it('passes flexible gpt-image-2 sizes to OpenAI and tracks returned token usage', async () => {
        const provider = new OpenAIProvider('sk-test');
        const generate = vi.fn().mockResolvedValue({
            data: [{ b64_json: 'YWJjMTIz' }],
            usage: {
                input_tokens: 150,
                input_tokens_details: {
                    text_tokens: 100,
                    image_tokens: 50,
                },
                output_tokens: 5500,
                total_tokens: 5650,
            },
        });

        (provider as any)._client = {
            images: {
                generate,
            },
        };

        const images = await provider.createImage(
            'A polished product render on a clean studio backdrop',
            'gpt-image-2',
            { agent_id: 'test-gpt-image-2' } as any,
            {
                quality: 'high',
                size: '2048x1152',
            }
        );

        expect(images).toEqual(['data:image/png;base64,YWJjMTIz']);
        expect(generate).toHaveBeenCalledWith({
            model: 'gpt-image-2',
            prompt: 'A polished product render on a clean studio backdrop',
            n: 1,
            background: 'auto',
            quality: 'high',
            size: '2048x1152',
            moderation: 'low',
            output_format: 'png',
        });

        const expectedCost = (100 / 1_000_000) * 5 + (50 / 1_000_000) * 8 + (5500 / 1_000_000) * 30;
        expect(costTracker.getTotalCost()).toBeCloseTo(expectedCost);
        expect(costTracker.getCostsByModel()['gpt-image-2']?.calls).toBe(1);
    });

    it('uses documented common-size estimates when OpenAI does not return usage', async () => {
        const provider = new OpenAIProvider('sk-test');
        const generate = vi.fn().mockResolvedValue({
            data: [{ b64_json: 'YWJjMTIz' }],
        });

        (provider as any)._client = {
            images: {
                generate,
            },
        };

        await provider.createImage(
            'A clean architectural poster',
            'gpt-image-2',
            { agent_id: 'test-gpt-image-2-estimate' } as any,
            {
                quality: 'medium',
                size: '1536x1024',
            }
        );

        expect(costTracker.getCostsByModel()['gpt-image-2']?.cost).toBeCloseTo(0.041);
    });

    it('estimates gpt-image-2 pricing for custom sizes when OpenAI does not return usage', async () => {
        const provider = new OpenAIProvider('sk-test');
        const addUsageSpy = vi.spyOn(costTracker, 'addUsage');
        const generate = vi.fn().mockResolvedValue({
            data: [{ b64_json: 'YWJjMTIz' }],
        });

        (provider as any)._client = {
            images: {
                generate,
            },
        };

        await provider.createImage(
            'A clean architectural poster',
            'gpt-image-2',
            { agent_id: 'test-gpt-image-2-custom-estimate' } as any,
            {
                quality: 'medium',
                size: '2048x1152',
            }
        );

        expect(costTracker.getCostsByModel()['gpt-image-2']?.cost).toBeCloseTo(0.0615);
        expect(addUsageSpy.mock.calls[0]?.[0]).toMatchObject({
            model: 'gpt-image-2',
            image_count: 1,
            cost: 0.0615,
            metadata: {
                quality: 'medium',
                size: '2048x1152',
                pricing_source: 'ensemble_size_estimate',
                estimated_output_tokens: 2050,
                output_price_per_million: 30,
                cost_per_image: 0.0615,
                estimated: true,
            },
        });
    });
});
