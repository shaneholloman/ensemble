import { describe, expect, it } from 'vitest';
import type { AgentDefinition, ProviderStreamEvent, ResponseInput } from '../types/types.js';
import { setEnsembleRequestFunction, verifyOutput } from '../utils/verification.js';

function createVerifierStream(response: string) {
    return async function* (_messages: ResponseInput, agent: AgentDefinition): AsyncGenerator<ProviderStreamEvent> {
        expect(agent.modelSettings?.json_schema?.strict).toBe(true);
        yield {
            type: 'message_complete',
            message_id: 'verifier-message',
            content: response,
        } as ProviderStreamEvent;
    };
}

describe('verifyOutput', () => {
    it('returns the parsed verifier result when the strict schema is satisfied', async () => {
        setEnsembleRequestFunction(createVerifierStream('{"status":"pass"}'));

        await expect(verifyOutput({ model: 'test-model' }, 'candidate output', [])).resolves.toEqual({
            status: 'pass',
        });
    });

    it('rejects verifier responses that violate the authoritative schema', async () => {
        setEnsembleRequestFunction(createVerifierStream('{"status":"maybe"}'));

        const result = await verifyOutput({ model: 'test-model' }, 'candidate output', []);

        expect(result.status).toBe('fail');
        expect(result.reason).toContain('Structured output failed schema validation');
    });
});
