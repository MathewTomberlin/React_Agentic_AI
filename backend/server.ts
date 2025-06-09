#!/usr/bin/env node
const express = require('express')
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import axios from 'axios';

//SwarmUI MCP Server
class SwarmUIServer {
    private server: Server;
    private axiosInstance;
    private sessionId: string | null = null;

    constructor() {
        console.error('[Setup] Initializing SwarmUI MCP server...');

        //Describe the server and the available tools. These descriptions will be used by llm agents to understand what the server can do.
        this.server = new Server(
            {
                name: 'swarmui-mcp-server',
                version: '0.1.0',
            },
            {
                capabilities: {
                    tools: {
                        generateImage: {
                            description: 'Generate an image from a positive prompt and optional negative prompt using SwarmUI. Other generation parameters such as number of images, steps, and cfg may be supplied.',
                            inputSchema: {
                                type: 'object',
                                properties: {
                                    prompt: { type: 'string', description: 'The prompt to generate an image from.' },
                                    negative: { type: 'string', description: 'Negative prompt to avoid certain features in the image.' },
                                    images: { type: 'number', description: 'Number of images to generate' },
                                    steps: { type: 'number', description: 'Number of steps for image generation' },
                                    cfgScale: { type: 'number', description: 'Classifier-free guidance scale' },
                                    seed: { type: 'number', description: 'Seed for random number generation, -1 for random' },
                                },
                                required: ['prompt']
                            },
                            outputSchema: {
                                type: 'object',
                                properties: {
                                    images: { type: 'array', items: { type: 'string' }, description: 'Array of image URLs.' }
                                }
                            }
                        }
                    },
                },
            }
        );

        // Define the axios instance for making HTTP requests to the SwarmUI API
        this.axiosInstance = axios.create({
            baseURL: 'http://localhost:7801/API',
            timeout: 300000,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });
    }


    // Get a new session ID from the SwarmUI API
    private async getSessionId(): Promise<string> {
        const response = await this.axiosInstance.post('/GetNewSession', {});
        return response.data.session_id;
    }
    //prompt: '(masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, ultra detailed, 8k, detailed background),' + prompt + 'subsurface scattering, sharp focus, film grain, lens flare, depth of field, 35mm photograph, accurate anatomy, natural proportions, detailed textures, skindentation',
    //negative: ''(flat_color: 1.4), (anime_coloring: 1.4), (anime_style: 1.4), (lowres, low_res: 1.4), (low_detail: 1.4), (low_poly: 1.4), (big_eyes: 1.4), (line_shading: 1.4), (sketch, lineart, outline, thick_outline), (traditional_media), quality, watermark, comic, vector_trace, blurry, bokeh, censored, bad_3d, fewer_digits, bad_anatomy, bad_feet, bad_hands, bad_proportions, bad_perspective, shiny_skin, smooth_skin, unfinished, artistic_error, lossless - lossy' + negative'
    // Generate an image using the SwarmUI API with the provided parameters
    async generateImage(prompt: string, negative: string, images: number, seed: number, steps: number, cfgScale: number, model: string, width: number, height: number, refinerControlPercentage: number, refinerUpscale: number, refinerMethod:string, sampler:string, scheduler:string) {
        if (!this.sessionId) throw new Error('[Error] Session ID not initialized');
        const request = {
            session_id: this.sessionId,
            seed,
            images,
            steps,
            cfgScale,
            prompt: prompt,
            negativePrompt: negative,
            model,
            sampler,
            scheduler,
            width,
            height,
            refinerControlPercentage,
            refinerUpscale,
            refinerMethod,
        }
        const response = await this.axiosInstance.post('/GenerateText2Image', request);
        return response.data;
    }

    // Connect the server to the transport layer
    async run() {
        this.sessionId = await this.getSessionId();
        const transport = new StdioServerTransport();
        await this.server.connect(transport);
    }
}

// Create an Express server to handle HTTP requests
const app = express();
app.use(express.json());

// Initialize a single SwarmUI server instance
let swarmServer: SwarmUIServer | null = null;

// Initialize the server before handling any requests
async function initializeServer() {
    try {
        swarmServer = new SwarmUIServer();
        await swarmServer.run();
        console.log('[Setup] SwarmUI MCP server initialized and running');
    } catch (err) {
        console.error('[Error] Failed to initialize SwarmUI server:', err);
        throw err;
    }
}

// Endpoint to generate an image using the SwarmUI MCP server
app.post('/generate-image', async (req, res) => {
    try {
        // Ensure server is initialized
        if (!swarmServer) {
            throw new Error('[Error] SwarmUI server not initialized');
        }
        const prompt = String(req.body.prompt || "");
        const negative = String(req.body.negative || "");
        const images = parseInt(req.body.images || "1", 10);
        const steps = parseInt(req.body.steps || "50", 10);
        const cfgScale = parseFloat(req.body.cfgScale || "3.0");
        const seed = parseInt(req.body.seed || "-1", 10);
        const model = String(req.body.model || "hyphoriaIllustrious20_v001");
        const width = parseInt(req.body.width || "1024", 10);
        const height = parseInt(req.body.height || "1024", 10);
        const refinerControlPercentage = parseFloat(req.body.refinerControlPercentage || "0.0");
        const refinerUpscale = parseFloat(req.body.refinerUpscale || "2.0");
        const refinerMethod = String(req.body.refinerMethod || "stepSwap");
        const sampler = String(req.body.sampler || "dpmpp_3m_sde_gpu");
        const scheduler = String(req.body.scheduler || "simple");

        console.log(`[Request] Generate Image\nprompt: ${prompt}\nnegative: ${negative}\nimages: ${images}\nsteps: ${steps}\ncfgScale: ${cfgScale}\nseed: ${seed}\nmodel: ${model}\nwidth: ${width}\nheight: ${height}\nrefinerControl: ${refinerControlPercentage}\nrefinerUpscale: ${refinerUpscale}\nrefinerMethod: ${refinerMethod}\nsampler: ${sampler}\nscheduler: ${scheduler}`);
        const result = await swarmServer.generateImage(prompt, negative, images, seed, steps, cfgScale,model,width,height,refinerControlPercentage,refinerUpscale,refinerMethod,sampler,scheduler);
        res.json(result);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Initialize server before starting to listen
(async () => {
    try {
        await initializeServer();
        app.listen(5001, () => {
            console.log('[Start] HTTP API for MCP server running on port 5001');
        });
    } catch (err) {
        console.error('[Error] Failed to start server:', err);
    }
})();