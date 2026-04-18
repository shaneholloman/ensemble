/**
 * Verification utilities for validating agent outputs
 */

import {
    ResponseInput,
    AgentDefinition,
    ResponseOutputMessage,
    ResponseInputMessage,
    ProviderStreamEvent,
} from '../types/types.js';
import { validateJsonResponseContent } from './json_schema.js';

export interface VerificationResult {
    status: 'pass' | 'fail';
    reason?: string;
}

// Type for ensemble request function to avoid circular dependency
type EnsembleRequestFunction = (messages: ResponseInput, agent: AgentDefinition) => AsyncGenerator<ProviderStreamEvent>;

// Global reference to ensemble request function (set by ensemble_request.ts)
let ensembleRequestFunction: EnsembleRequestFunction | null = null;

/**
 * Set the ensemble request function (called by ensemble_request.ts to avoid circular dependency)
 */
export function setEnsembleRequestFunction(fn: EnsembleRequestFunction): void {
    ensembleRequestFunction = fn;
}

/**
 * Verify an agent's output using a verifier agent
 */
export async function verifyOutput(
    verifier: AgentDefinition,
    output: string,
    originalMessages: ResponseInput
): Promise<VerificationResult> {
    const verificationSchema = {
        type: 'object',
        properties: {
            status: { type: 'string', enum: ['pass', 'fail'] },
            reason: { type: 'string' },
        },
        required: ['status'],
        additionalProperties: false,
    } as const;

    const verificationPrompt = `Please verify if the following output is correct and complete:

${output}

Respond with JSON: {"status": "pass"} or {"status": "fail", "reason": "explanation of what is wrong"}`;

    const verificationMessages: ResponseInput = [
        ...originalMessages,
        {
            type: 'message',
            role: 'assistant',
            content: output,
            status: 'completed',
        } as ResponseOutputMessage,
        {
            type: 'message',
            role: 'user',
            content: verificationPrompt,
        } as ResponseInputMessage,
    ];

    // Create a verifier with JSON schema enforcement
    const verifierWithSchema: AgentDefinition = {
        ...verifier,
        modelSettings: {
            ...verifier.modelSettings,
            json_schema: {
                type: 'json_schema',
                name: 'verification_result',
                strict: true,
                schema: verificationSchema,
            },
        },
    };

    if (!ensembleRequestFunction) {
        throw new Error('Ensemble request function not set. This is a circular dependency issue.');
    }

    try {
        const stream = ensembleRequestFunction(verificationMessages, verifierWithSchema);
        let fullResponse = '';

        for await (const event of stream) {
            if (event.type === 'message_complete' && 'content' in event) {
                fullResponse = event.content;
            }
        }

        const validation = validateJsonResponseContent(fullResponse, verificationSchema);
        if ('error' in validation) {
            return {
                status: 'fail',
                reason: validation.error,
            };
        }

        return validation.value as VerificationResult;
    } catch (error) {
        console.error('Verification failed:', error);
        return {
            status: 'fail',
            reason: error instanceof Error ? error.message : 'Invalid verification response',
        };
    }
}
