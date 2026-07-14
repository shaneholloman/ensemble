import { afterEach, describe, expect, it, vi } from 'vitest';
import { OpenAIProvider } from '../model_providers/openai.js';
import { normalizeOpenAIImageError, runOpenAIImageRequest } from '../model_providers/openai_image_request.js';

describe('OpenAI image request lifecycle', () => {
    afterEach(() => {
        vi.useRealTimers();
        vi.restoreAllMocks();
        vi.unstubAllGlobals();
    });

    it('settles at the hard deadline even when the provider promise ignores abort', async () => {
        vi.useFakeTimers();
        let providerSignal: AbortSignal | undefined;
        const request = runOpenAIImageRequest({
            operationName: 'OpenAI image edit with gpt-image-2',
            timeoutMs: 40_000,
            execute: signal => {
                providerSignal = signal;
                return new Promise<never>(() => undefined);
            },
        });
        const rejected = expect(request).rejects.toMatchObject({
            name: 'OpenAIImageRequestError',
            reason: 'timeout',
            retryable: true,
            code: 'ETIMEDOUT',
        });

        await vi.advanceTimersByTimeAsync(40_000);

        await rejected;
        expect(providerSignal?.aborted).toBe(true);
    });

    it('surfaces parent cancellation without waiting for the provider promise', async () => {
        const controller = new AbortController();
        const request = runOpenAIImageRequest({
            operationName: 'OpenAI image generation with gpt-image-2',
            timeoutMs: 40_000,
            abortSignal: controller.signal,
            execute: () => new Promise<never>(() => undefined),
        });
        const rejected = expect(request).rejects.toMatchObject({
            reason: 'aborted',
            retryable: false,
            code: 'ABORT_ERR',
        });

        controller.abort(new Error('conversion canceled'));

        await rejected;
    });

    it('preserves content-policy refusal details as a terminal provider error', () => {
        const error = Object.assign(new Error('Request rejected by the content policy.'), {
            status: 400,
            code: 'content_policy_violation',
            type: 'invalid_request_error',
            param: 'prompt',
            requestID: 'req_policy_123',
        });

        expect(normalizeOpenAIImageError(error)).toMatchObject({
            reason: 'content_policy_violation',
            retryable: false,
            status: 400,
            code: 'content_policy_violation',
            type: 'invalid_request_error',
            param: 'prompt',
            requestID: 'req_policy_123',
        });
    });

    it('propagates a provider refusal through the image editing API without retrying', async () => {
        const provider = new OpenAIProvider('sk-test');
        vi.spyOn(console, 'error').mockImplementation(() => undefined);
        const refusal = Object.assign(new Error('Request rejected by the content policy.'), {
            status: 400,
            code: 'content_policy_violation',
            type: 'invalid_request_error',
            requestID: 'req_policy_edit_123',
        });
        const withResponse = vi.fn().mockRejectedValue(refusal);
        const edit = vi.fn().mockReturnValue({ withResponse });
        (provider as any)._client = { images: { edit } };

        await expect(
            provider.createImage(
                'Repair only the transparent pixels.',
                'gpt-image-2',
                { agent_id: 'test-openai-image-refusal' } as any,
                {
                    source_images: ['data:image/png;base64,YWJjMTIz'],
                    quality: 'low',
                    timeout_ms: 40_000,
                }
            )
        ).rejects.toMatchObject({
            reason: 'content_policy_violation',
            retryable: false,
            status: 400,
            code: 'content_policy_violation',
            requestID: 'req_policy_edit_123',
        });
        expect(edit).toHaveBeenCalledTimes(1);
        expect(withResponse).toHaveBeenCalledTimes(1);
    });

    it('applies the hard deadline while a source image download is still preparing', async () => {
        vi.useFakeTimers();
        const provider = new OpenAIProvider('sk-test');
        const edit = vi.fn();
        (provider as any)._client = { images: { edit } };
        vi.stubGlobal(
            'fetch',
            vi.fn(() => new Promise<never>(() => undefined))
        );

        const request = provider.createImage(
            'Repair only the transparent pixels.',
            'gpt-image-2',
            { agent_id: 'test-openai-image-preparation-timeout' } as any,
            {
                source_images: ['https://example.com/non-settling.png'],
                quality: 'low',
                timeout_ms: 40_000,
            }
        );
        const rejected = expect(request).rejects.toMatchObject({
            reason: 'timeout',
            retryable: true,
            code: 'ETIMEDOUT',
        });

        await vi.advanceTimersByTimeAsync(40_000);

        await rejected;
        expect(edit).not.toHaveBeenCalled();
    });

    it('rejects a successful response with no image and retains its provider request id', async () => {
        const provider = new OpenAIProvider('sk-test');
        vi.spyOn(console, 'error').mockImplementation(() => undefined);
        const generate = vi.fn().mockReturnValue({
            withResponse: vi.fn().mockResolvedValue({
                data: { data: [] },
                request_id: 'req_empty_123',
            }),
        });
        (provider as any)._client = { images: { generate } };

        await expect(
            provider.createImage(
                'A simple geometric poster.',
                'gpt-image-2',
                { agent_id: 'test-openai-image-empty' } as any,
                { timeout_ms: 40_000 }
            )
        ).rejects.toMatchObject({
            reason: 'empty_response',
            retryable: false,
            requestID: 'req_empty_123',
        });
    });

    it('passes timeout, abort, and idempotency controls to the SDK and exposes the provider request id', async () => {
        const provider = new OpenAIProvider('sk-test');
        const onMetadata = vi.fn();
        const withResponse = vi.fn().mockResolvedValue({
            data: { data: [{ b64_json: 'YWJjMTIz' }] },
            request_id: 'req_provider_123',
        });
        const edit = vi.fn().mockReturnValue({ withResponse });
        (provider as any)._client = { images: { edit } };

        const images = await provider.createImage(
            'Repair only the transparent pixels.',
            'gpt-image-2',
            { agent_id: 'test-openai-image-lifecycle' } as any,
            {
                source_images: ['data:image/png;base64,YWJjMTIz'],
                mask: 'data:image/png;base64,YWJjMTIz',
                quality: 'low',
                timeout_ms: 40_000,
                request_id: 'conversion-effect-123',
                idempotency_key: 'stable-dispatch-123',
                on_metadata: onMetadata,
            }
        );

        expect(images).toEqual(['data:image/png;base64,YWJjMTIz']);
        expect(edit).toHaveBeenCalledWith(
            expect.objectContaining({
                model: 'gpt-image-2',
                moderation: 'low',
                output_format: 'png',
            }),
            expect.objectContaining({
                timeout: 40_000,
                maxRetries: 0,
                idempotencyKey: 'stable-dispatch-123',
                signal: expect.any(AbortSignal),
            })
        );
        expect(withResponse).toHaveBeenCalledTimes(1);
        expect(onMetadata).toHaveBeenCalledWith({
            model: 'gpt-image-2',
            provider: 'openai',
            provider_request_id: 'req_provider_123',
            request_id: 'conversion-effect-123',
        });
    });
});
