#!/usr/bin/env node
import 'dotenv/config';
import fs from 'node:fs/promises';
import path from 'node:path';
import { OpenAIProvider } from '../dist/model_providers/openai.js';

const sourcePath = process.env.OPENAI_IMAGE_SOURCE || process.argv[2];
const model = process.env.OPENAI_IMAGE_MODEL || 'gpt-image-2';
const quality = process.env.OPENAI_IMAGE_QUALITY || 'low';
const timeoutMs = Number(process.env.OPENAI_IMAGE_TIMEOUT_MS || 160_000);
const requestId = `ensemble-openai-image-smoke-${Date.now()}`;

const trace = event => console.log(JSON.stringify({ at: new Date().toISOString(), ...event }));

const combinedHeaders = (input, init) => {
    const headers = new Headers(input instanceof Request ? input.headers : undefined);
    new Headers(init?.headers).forEach((value, name) => headers.set(name, value));
    return headers;
};

const inputUrl = input => {
    if (typeof input === 'string') return input;
    if (input instanceof URL) return input.href;
    return input.url;
};

const installSanitizedFetchTrace = () => {
    const realFetch = globalThis.fetch;
    let attempt = 0;
    globalThis.fetch = async (input, init) => {
        const url = new URL(inputUrl(input));
        if (url.hostname !== 'api.openai.com' || !url.pathname.startsWith('/v1/images/')) {
            return await realFetch(input, init);
        }

        attempt += 1;
        const headers = combinedHeaders(input, init);
        const startedAt = performance.now();
        trace({
            event: 'request.started',
            attempt,
            method: init?.method || (input instanceof Request ? input.method : 'GET'),
            endpoint: `${url.origin}${url.pathname}`,
            content_type: headers.get('content-type')?.split(';')[0],
            content_length: headers.get('content-length'),
            idempotency_key: headers.get('idempotency-key'),
            sdk_retry_count: headers.get('x-stainless-retry-count'),
        });

        try {
            const response = await realFetch(input, init);
            trace({
                event: 'request.completed',
                attempt,
                status: response.status,
                elapsed_ms: Math.round(performance.now() - startedAt),
                provider_request_id: response.headers.get('x-request-id'),
                provider_processing_ms: response.headers.get('openai-processing-ms'),
            });
            return response;
        } catch (error) {
            trace({
                event: 'request.failed',
                attempt,
                elapsed_ms: Math.round(performance.now() - startedAt),
                error_name: error instanceof Error ? error.name : typeof error,
                error_message: error instanceof Error ? error.message : String(error),
            });
            throw error;
        }
    };
};

const imageDataUrl = async filePath => {
    const absolutePath = path.resolve(filePath);
    const extension = path.extname(absolutePath).toLowerCase();
    const mime = extension === '.jpg' || extension === '.jpeg' ? 'image/jpeg' : 'image/png';
    const bytes = await fs.readFile(absolutePath);
    trace({ event: 'source.loaded', mime, byte_count: bytes.byteLength });
    return `data:${mime};base64,${bytes.toString('base64')}`;
};

const normalizedError = error => ({
    name: error?.name,
    message: error?.message,
    reason: error?.reason,
    retryable: error?.retryable,
    status: error?.status,
    code: error?.code,
    type: error?.type,
    param: error?.param,
    provider_request_id: error?.requestID || error?.request_id,
});

async function main() {
    if (!process.env.OPENAI_API_KEY) throw new Error('Missing OPENAI_API_KEY in environment.');
    if (!sourcePath) throw new Error('Pass a source image path or set OPENAI_IMAGE_SOURCE.');
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('OPENAI_IMAGE_TIMEOUT_MS must be positive.');

    installSanitizedFetchTrace();
    const provider = new OpenAIProvider(process.env.OPENAI_API_KEY);
    const source = await imageDataUrl(sourcePath);
    let metadata;
    const startedAt = performance.now();

    const images = await provider.createImage(
        'Preserve this image exactly; make no visible changes.',
        model,
        { agent_id: 'openai-image-edit-smoke', tags: ['smoke', 'image-edit'] },
        {
            source_images: [source],
            n: 1,
            quality,
            timeout_ms: timeoutMs,
            request_id: requestId,
            idempotency_key: requestId,
            on_metadata: value => {
                metadata = value;
                trace({ event: 'metadata.received', ...value });
            },
        }
    );

    const base64 = images[0]?.split(',', 2)[1] || '';
    trace({
        event: 'smoke.completed',
        model,
        quality,
        request_id: requestId,
        elapsed_ms: Math.round(performance.now() - startedAt),
        image_count: images.length,
        returned_byte_count: Buffer.from(base64, 'base64').byteLength,
        provider_request_id: metadata?.provider_request_id,
    });
}

main().catch(error => {
    trace({ event: 'smoke.failed', ...normalizedError(error) });
    process.exitCode = 1;
});
