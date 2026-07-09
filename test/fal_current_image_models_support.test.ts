import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { findModel } from '../data/model_data.js';
import { FALProvider } from '../model_providers/fal.js';
import { getModelFromAgent, getModelProvider, getProviderFromModel } from '../model_providers/model_provider.js';
import { costTracker } from '../utils/cost_tracker.js';

const originalFalKey = process.env.FAL_KEY;

describe('current FAL image model support', () => {
    beforeEach(() => {
        process.env.FAL_KEY = 'fal-test';
        costTracker.reset();
    });

    afterEach(() => {
        process.env.FAL_KEY = originalFalKey;
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
    });

    it('registers current Seedream 5 and Ideogram V4 models on the FAL provider', async () => {
        expect(findModel('bytedance/seedream/v5/pro')).toMatchObject({
            id: 'bytedance/seedream/v5/pro/text-to-image',
            provider: 'fal',
            cost: { per_image: 0.135 },
            features: { input_modality: ['text'], output_modality: ['image'] },
            class: 'image_generation',
        });
        expect(findModel('bytedance/seedream/v5/lite')).toMatchObject({
            id: 'fal-ai/bytedance/seedream/v5/lite/text-to-image',
            provider: 'fal',
            cost: { per_image: 0.035 },
            features: { input_modality: ['text'], output_modality: ['image'] },
            class: 'image_generation',
        });
        expect(findModel('ideogram/v4/instant')).toMatchObject({
            id: 'ideogram/v4/instant',
            provider: 'fal',
            cost: { per_image: 0.0075 },
            features: { input_modality: ['text'], output_modality: ['image'] },
            class: 'image_generation',
        });
        expect(findModel('ideogram/v4/fast')).toMatchObject({
            id: 'ideogram/v4/fast',
            provider: 'fal',
            cost: { per_image: 0.0105 },
            features: { input_modality: ['text'], output_modality: ['image'] },
            class: 'image_generation',
        });

        expect(getProviderFromModel('bytedance/seedream/v5/pro')).toBe('fal');
        expect(getProviderFromModel('bytedance/seedream/v5/lite')).toBe('fal');
        expect(getProviderFromModel('ideogram/v4/instant')).toBe('fal');
        expect(getProviderFromModel('ideogram/v4/fast')).toBe('fal');
        expect(getModelProvider('ideogram/v4/fast')).toBeInstanceOf(FALProvider);
        expect(
            await getModelFromAgent({ agent_id: 'test-seedream-v5-pro', model: 'bytedance/seedream/v5/pro' } as any)
        ).toBe('bytedance/seedream/v5/pro/text-to-image');
        expect(
            await getModelFromAgent({ agent_id: 'test-seedream-v5-lite', model: 'bytedance/seedream/v5/lite' } as any)
        ).toBe('fal-ai/bytedance/seedream/v5/lite/text-to-image');
    });

    it('calls Seedream 5 Pro text-to-image with FAL request options and tiered image pricing', async () => {
        const provider = new FALProvider();
        const fetchMock = vi.fn().mockResolvedValue(
            new Response(
                JSON.stringify({
                    images: [
                        { url: 'https://example.com/small.png', width: 1536, height: 1536 },
                        { url: 'https://example.com/large.png', width: 2048, height: 2048 },
                    ],
                }),
                { status: 200, headers: { 'Content-Type': 'application/json' } }
            )
        );
        vi.stubGlobal('fetch', fetchMock);

        const images = await provider.createImage(
            'editorial product poster',
            'bytedance/seedream/v5/pro',
            { agent_id: 'test-seedream-v5-pro' } as any,
            {
                n: 2,
                resolution: '1k',
                output_format: 'png',
                enable_safety_checker: false,
                response_format: 'b64_json',
                request_id: 'seedream-v5-pro-request',
            }
        );

        expect(images).toEqual(['https://example.com/small.png', 'https://example.com/large.png']);
        expect(fetchMock).toHaveBeenCalledWith('https://fal.run/bytedance/seedream/v5/pro/text-to-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: 'Key fal-test',
            },
            body: JSON.stringify({
                prompt: 'editorial product poster',
                image_size: 'auto_1K',
                num_images: 2,
                output_format: 'png',
                sync_mode: true,
                enable_safety_checker: false,
            }),
        });
        expect(costTracker.getTotalCost()).toBeCloseTo(0.2025);
    });

    it('calls Seedream 5 Lite text-to-image with the documented FAL endpoint and flat image pricing', async () => {
        const provider = new FALProvider();
        const fetchMock = vi.fn().mockResolvedValue(
            new Response(
                JSON.stringify({
                    images: [
                        { url: 'https://example.com/lite-a.png', width: 3072, height: 3072 },
                        { url: 'https://example.com/lite-b.png', width: 3072, height: 3072 },
                    ],
                }),
                { status: 200, headers: { 'Content-Type': 'application/json' } }
            )
        );
        vi.stubGlobal('fetch', fetchMock);

        const images = await provider.createImage(
            'product campaign image',
            'bytedance/seedream/v5/lite',
            { agent_id: 'test-seedream-v5-lite' } as any,
            {
                n: 2,
                size: '2048x2048',
                enable_safety_checker: true,
                response_format: 'b64_json',
                request_id: 'seedream-v5-lite-request',
            }
        );

        expect(images).toEqual(['https://example.com/lite-a.png', 'https://example.com/lite-b.png']);
        expect(fetchMock).toHaveBeenCalledWith('https://fal.run/fal-ai/bytedance/seedream/v5/lite/text-to-image', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: 'Key fal-test',
            },
            body: JSON.stringify({
                prompt: 'product campaign image',
                image_size: { width: 2048, height: 2048 },
                num_images: 2,
                sync_mode: true,
                enable_safety_checker: true,
            }),
        });
        expect(costTracker.getTotalCost()).toBeCloseTo(0.07);
    });

    it('calls Ideogram V4 instant with the FAL endpoint and default balanced megapixel pricing', async () => {
        const provider = new FALProvider();
        const fetchMock = vi.fn().mockResolvedValue(
            new Response(JSON.stringify({ images: [{ url: 'https://example.com/instant.jpg' }] }), {
                status: 200,
                headers: { 'Content-Type': 'application/json' },
            })
        );
        vi.stubGlobal('fetch', fetchMock);

        const images = await provider.createImage(
            'badge with crisp lettering',
            'ideogram/v4/instant',
            { agent_id: 'test-ideogram-v4-instant' } as any,
            {
                n: 1,
                size: '2048x2048',
                output_format: 'jpeg',
                request_id: 'ideogram-v4-instant-request',
            }
        );

        expect(images).toEqual(['https://example.com/instant.jpg']);
        expect(fetchMock).toHaveBeenCalledWith('https://fal.run/ideogram/v4/instant', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: 'Key fal-test',
            },
            body: JSON.stringify({
                prompt: 'badge with crisp lettering',
                num_images: 1,
                image_size: { width: 2048, height: 2048 },
                output_format: 'jpeg',
            }),
        });
        expect(costTracker.getTotalCost()).toBeCloseTo(0.03);
    });

    it('calls Ideogram V4 fast with rendering speed and quality-tier megapixel pricing', async () => {
        const provider = new FALProvider();
        const fetchMock = vi.fn().mockResolvedValue(
            new Response(
                JSON.stringify({ images: [{ url: 'https://example.com/fast.jpg', width: 2048, height: 2048 }] }),
                {
                    status: 200,
                    headers: { 'Content-Type': 'application/json' },
                }
            )
        );
        vi.stubGlobal('fetch', fetchMock);

        const images = await provider.createImage(
            'poster with crisp lettering',
            'ideogram/v4/fast',
            { agent_id: 'test-ideogram-v4-fast' } as any,
            {
                size: 'square',
                quality: 'high',
                seed: 9.8,
                enable_safety_checker: true,
                response_format: 'b64_json',
                request_id: 'ideogram-v4-fast-request',
            }
        );

        expect(images).toEqual(['https://example.com/fast.jpg']);
        expect(fetchMock).toHaveBeenCalledWith('https://fal.run/ideogram/v4/fast', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                Authorization: 'Key fal-test',
            },
            body: JSON.stringify({
                prompt: 'poster with crisp lettering',
                num_images: 1,
                image_size: 'square_hd',
                rendering_speed: 'QUALITY',
                sync_mode: true,
                enable_safety_checker: true,
                seed: 9,
            }),
        });
        expect(costTracker.getTotalCost()).toBeCloseTo(0.07);
    });
});
