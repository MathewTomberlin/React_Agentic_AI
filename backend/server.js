#!/usr/bin/env node
"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var express = require('express');
var index_js_1 = require("@modelcontextprotocol/sdk/server/index.js");
var stdio_js_1 = require("@modelcontextprotocol/sdk/server/stdio.js");
var axios_1 = require("axios");
//SwarmUI MCP Server
var SwarmUIServer = /** @class */ (function () {
    function SwarmUIServer() {
        this.sessionId = null;
        console.error('[Setup] Initializing SwarmUI MCP server...');
        //Describe the server and the available tools. These descriptions will be used by llm agents to understand what the server can do.
        this.server = new index_js_1.Server({
            name: 'swarmui-mcp-server',
            version: '0.1.0',
        }, {
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
        });
        // Define the axios instance for making HTTP requests to the SwarmUI API
        this.axiosInstance = axios_1.default.create({
            baseURL: 'http://localhost:7801/API',
            timeout: 300000,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            }
        });
    }
    // Get a new session ID from the SwarmUI API
    SwarmUIServer.prototype.getSessionId = function () {
        return __awaiter(this, void 0, void 0, function () {
            var response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.axiosInstance.post('/GetNewSession', {})];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.data.session_id];
                }
            });
        });
    };
    //prompt: '(masterpiece, best quality, amazing quality, very aesthetic, absurdres, newest, ultra detailed, 8k, detailed background),' + prompt + 'subsurface scattering, sharp focus, film grain, lens flare, depth of field, 35mm photograph, accurate anatomy, natural proportions, detailed textures, skindentation',
    //negative: ''(flat_color: 1.4), (anime_coloring: 1.4), (anime_style: 1.4), (lowres, low_res: 1.4), (low_detail: 1.4), (low_poly: 1.4), (big_eyes: 1.4), (line_shading: 1.4), (sketch, lineart, outline, thick_outline), (traditional_media), quality, watermark, comic, vector_trace, blurry, bokeh, censored, bad_3d, fewer_digits, bad_anatomy, bad_feet, bad_hands, bad_proportions, bad_perspective, shiny_skin, smooth_skin, unfinished, artistic_error, lossless - lossy' + negative'
    // Generate an image using the SwarmUI API with the provided parameters
    SwarmUIServer.prototype.generateImage = function (prompt, negative, images, seed, steps, cfgScale, model, width, height, refinerControlPercentage, refinerUpscale, refinerMethod, sampler, scheduler) {
        return __awaiter(this, void 0, void 0, function () {
            var request, response;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        if (!this.sessionId)
                            throw new Error('[Error] Session ID not initialized');
                        request = {
                            session_id: this.sessionId,
                            seed: seed,
                            images: images,
                            steps: steps,
                            cfgScale: cfgScale,
                            prompt: prompt,
                            negativePrompt: negative,
                            model: model,
                            sampler: sampler,
                            scheduler: scheduler,
                            width: width,
                            height: height,
                            refinerControlPercentage: refinerControlPercentage,
                            refinerUpscale: refinerUpscale,
                            refinerMethod: refinerMethod,
                        };
                        return [4 /*yield*/, this.axiosInstance.post('/GenerateText2Image', request)];
                    case 1:
                        response = _a.sent();
                        return [2 /*return*/, response.data];
                }
            });
        });
    };
    // Connect the server to the transport layer
    SwarmUIServer.prototype.run = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _a, transport;
            return __generator(this, function (_b) {
                switch (_b.label) {
                    case 0:
                        _a = this;
                        return [4 /*yield*/, this.getSessionId()];
                    case 1:
                        _a.sessionId = _b.sent();
                        transport = new stdio_js_1.StdioServerTransport();
                        return [4 /*yield*/, this.server.connect(transport)];
                    case 2:
                        _b.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    return SwarmUIServer;
}());
// Create an Express server to handle HTTP requests
var app = express();
app.use(express.json());
// Initialize a single SwarmUI server instance
var swarmServer = null;
// Initialize the server before handling any requests
function initializeServer() {
    return __awaiter(this, void 0, void 0, function () {
        var err_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 2, , 3]);
                    swarmServer = new SwarmUIServer();
                    return [4 /*yield*/, swarmServer.run()];
                case 1:
                    _a.sent();
                    console.log('[Setup] SwarmUI MCP server initialized and running');
                    return [3 /*break*/, 3];
                case 2:
                    err_1 = _a.sent();
                    console.error('[Error] Failed to initialize SwarmUI server:', err_1);
                    throw err_1;
                case 3: return [2 /*return*/];
            }
        });
    });
}
// Endpoint to generate an image using the SwarmUI MCP server
app.post('/generate-image', function (req, res) { return __awaiter(void 0, void 0, void 0, function () {
    var prompt_1, negative, images, steps, cfgScale, seed, model, width, height, refinerControlPercentage, refinerUpscale, refinerMethod, sampler, scheduler, result, err_2;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                // Ensure server is initialized
                if (!swarmServer) {
                    throw new Error('[Error] SwarmUI server not initialized');
                }
                prompt_1 = String(req.body.prompt || "");
                negative = String(req.body.negative || "");
                images = parseInt(req.body.images || "1", 10);
                steps = parseInt(req.body.steps || "50", 10);
                cfgScale = parseFloat(req.body.cfgScale || "3.0");
                seed = parseInt(req.body.seed || "-1", 10);
                model = String(req.body.model || "hyphoriaIllustrious20_v001");
                width = parseInt(req.body.width || "1024", 10);
                height = parseInt(req.body.height || "1024", 10);
                refinerControlPercentage = parseFloat(req.body.refinerControlPercentage || "0.0");
                refinerUpscale = parseFloat(req.body.refinerUpscale || "2.0");
                refinerMethod = String(req.body.refinerMethod || "stepSwap");
                sampler = String(req.body.sampler || "dpmpp_3m_sde_gpu");
                scheduler = String(req.body.scheduler || "simple");
                console.log("[Request] Generate Image\nprompt: ".concat(prompt_1, "\nnegative: ").concat(negative, "\nimages: ").concat(images, "\nsteps: ").concat(steps, "\ncfgScale: ").concat(cfgScale, "\nseed: ").concat(seed, "\nmodel: ").concat(model, "\nwidth: ").concat(width, "\nheight: ").concat(height, "\nrefinerControl: ").concat(refinerControlPercentage, "\nrefinerUpscale: ").concat(refinerUpscale, "\nrefinerMethod: ").concat(refinerMethod, "\nsampler: ").concat(sampler, "\nscheduler: ").concat(scheduler));
                return [4 /*yield*/, swarmServer.generateImage(prompt_1, negative, images, seed, steps, cfgScale, model, width, height, refinerControlPercentage, refinerUpscale, refinerMethod, sampler, scheduler)];
            case 1:
                result = _a.sent();
                res.json(result);
                return [3 /*break*/, 3];
            case 2:
                err_2 = _a.sent();
                res.status(500).json({ error: err_2.message });
                return [3 /*break*/, 3];
            case 3: return [2 /*return*/];
        }
    });
}); });
// Initialize server before starting to listen
(function () { return __awaiter(void 0, void 0, void 0, function () {
    var err_3;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                _a.trys.push([0, 2, , 3]);
                return [4 /*yield*/, initializeServer()];
            case 1:
                _a.sent();
                app.listen(5001, function () {
                    console.log('[Start] HTTP API for MCP server running on port 5001');
                });
                return [3 /*break*/, 3];
            case 2:
                err_3 = _a.sent();
                console.error('[Error] Failed to start server:', err_3);
                return [3 /*break*/, 3];
            case 3: return [2 /*return*/];
        }
    });
}); })();
