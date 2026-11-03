# I2Sonic Worker — Phase 4 (voice in, voice out)

The cloud "brain" that replaces the laptop: Whisper hears, Gemini (as Friday) answers, a free voice speaks.

```
GET  /health                                          -> {"ok": true, "voices": ["melotts","aura","gemini"], ...}
POST /chat   {"text": "..."}  + X-Device-Token        -> {"reply": "...", "ms": 640}
POST /tts    {"text": "..."}  + X-Device-Token        -> audio/mpeg or audio/wav   (laptop testing)
POST /tts?provider=aura                               -> force one voice: melotts | aura | gemini
POST /say    {"text": "..."}  + X-Device-Token        -> reply text + audio        (S3, typed questions)
POST /ask    16 kHz PCM or WAV + X-Device-Token      -> heard + reply + audio     (S3, push-to-talk)
POST /ask?debug=1                                     -> {"heard", "reply", ...}   JSON only, no voice (laptop testing)
```

In the Friday repo this folder is `worker/`.

## Voices

Tried in the order of `TTS_PROVIDERS` in `wrangler.toml`; if one fails (quota, error) the next is used.

| Provider | Free allowance | Format | Notes |
|---|---|---|---|
| `melotts` (default) | ~500 min of speech / day (Workers AI, 10,000 neurons) | MP3 | The S3 decodes it. No extra API key |
| `aura` | ~50 replies / day (same neuron budget) | 16 kHz PCM | Nicer voice. `AURA_SPEAKER` picks it |
| `gemini` | ~10 / day | 16 kHz PCM | Last resort |

If all three fail, `/say` returns the text in `reply` and the S3 speaks it with its own offline voice (SAM).

## Hearing (speech to text)

Tried in the order of `STT_PROVIDERS` in `wrangler.toml`.

| Provider | Free allowance | Notes |
|---|---|---|
| `workers` (default) | ~200 min of questions / day (Workers AI Whisper large-v3-turbo, ~4 neurons per 5 s question) | No extra key. Shares the 10,000-neuron budget with the voices |
| `groq` | ~2,000 requests / day | Optional backup: `npx wrangler secret put GROQ_API_KEY` (free key at console.groq.com) |

Silence and Whisper's usual hallucinations ("Thanks for watching!", "you") count as nothing heard:
Friday says *"Sorry, I didn't catch that"* and Gemini isn't called. Recordings must be 0.3–30 s.

## Phase 4: deploy and test hearing

```powershell
npm run deploy
npx wrangler secret put GROQ_API_KEY        # optional backup, paste the key when asked
$u = "https://i2sonic.YOUR-SUBDOMAIN.workers.dev"
$t = (Select-String -Path .dev.vars -Pattern '^DEVICE_TOKEN=(.*)$').Matches[0].Groups[1].Value.Trim().Trim('"')
$h = @{ "X-Device-Token" = $t }
Invoke-RestMethod -Method Post -Uri "$u/ask?debug=1" -Headers $h -ContentType "application/octet-stream" -InFile ..\server\last_question.wav
```

Any 16-bit WAV of you asking something works (the laptop's `last_question.wav` is one); it must be WAV, not `.m4a`.

| # | Test | Expected |
|---|---|---|
| 1 | `$u/health` | `"stt":["workers","groq"]`, `"groq_key":true` if you set it |
| 2 | The `/ask?debug=1` call | `heard` = what you said, `reply` = Friday's answer, `stt_provider: workers` |
| 3 | Same call with a WAV of silence | `heard` empty, reply "Sorry, I didn't catch that..." |

| Error | Meaning |
|---|---|
| `400 audio too short` / `audio longer than 30 s` | Hold BOOT 0.3–30 s |
| `400 WAV is 24-bit, expected 16` | Re-export the WAV as 16-bit PCM |
| `heard` is nonsense or empty for a real question | The file isn't WAV (e.g. `.m4a`): it was read as raw audio |
| `500/502 hearing: ...` with `tried: [...]` | Whisper failed on every provider; the list says why each one did |

## Phase 3.2: deploy and test voices

```powershell
npm run deploy
$u = "https://i2sonic.YOUR-SUBDOMAIN.workers.dev"
$t = (Select-String -Path .dev.vars -Pattern '^DEVICE_TOKEN=(.*)$').Matches[0].Groups[1].Value.Trim().Trim('"')
$h = @{ "X-Device-Token" = $t }
$b = '{"text":"Hello, I am Jarvis. How can I help?"}'
Invoke-WebRequest -Method Post -Uri "$u/tts"               -Headers $h -ContentType "application/json" -Body $b -OutFile melo.mp3
Invoke-WebRequest -Method Post -Uri "$u/tts?provider=aura" -Headers $h -ContentType "application/json" -Body $b -OutFile aura.wav
start melo.mp3
start aura.wav
```

| # | Test | Expected |
|---|---|---|
| 1 | `$u/health` | `"voices":["melotts","aura","gemini"]`, `"workers_ai":true` |
| 2 | `melo.mp3` plays | MeloTTS voice |
| 3 | `aura.wav` plays | Aura voice (compare, and reorder `TTS_PROVIDERS` if you prefer it) |
| 4 | `npm run logs` during the tests | `X-TTS-Provider` and CPU time a few ms |

| Error | Meaning |
|---|---|
| `500 Workers AI binding missing` | `wrangler.toml` lost its `[ai]` section, or the deploy didn't pick it up |
| `429` with `tried: [...]` | Every voice in the chain is out of quota for today |
| `502 voice provider returned an unknown audio format` | MeloTTS changed its response: send me the log |

---

## 1. One-time setup (≈10 min)

| What | Where |
|---|---|
| Node.js 20+ | <https://nodejs.org> (LTS installer). Check: `node -v` |
| Cloudflare account (free) | <https://dash.cloudflare.com/sign-up> |
| Gemini API key (free) | <https://aistudio.google.com/apikey> |
| Your free-tier limits | <https://aistudio.google.com/rate-limit> — check `gemini-3.5-flash-lite` |

Then, in PowerShell inside this folder:

```powershell
npm install
npx wrangler login          # opens the browser, click Allow
```

Make up a device token: any long random string, e.g. from
`[guid]::NewGuid().ToString("N")` in PowerShell. Keep it; the S3 will use it.

## 2. Test locally first

```powershell
copy .dev.vars.example .dev.vars    # then open .dev.vars and paste your key + token
npm run dev                         # serves on http://localhost:8787
```

In a **second** PowerShell window:

```powershell
$h = @{ "X-Device-Token" = "YOUR-TOKEN" }
Invoke-RestMethod -Method Post -Uri http://localhost:8787/chat -Headers $h `
  -ContentType "application/json" -Body '{"text":"what is the capital of France"}'
```

## 3. Deploy

```powershell
npx wrangler secret put GEMINI_API_KEY    # paste key when asked
npx wrangler secret put DEVICE_TOKEN      # paste the same token
npm run deploy
```

It prints your URL, like `https://i2sonic.<your-subdomain>.workers.dev`.
Open `<url>/health` in a browser: you should see `{"ok":true,...}`.
Then repeat the PowerShell test above with that URL instead of localhost.

## Phase 1 pass checklist

| # | Test | Expected |
|---|---|---|
| 1 | `/health` in browser | `{"ok":true,"model":"gemini-3.5-flash-lite"}` |
| 2 | `/chat` with right token | short plain-English `reply`, `ms` under ~2000 |
| 3 | `/chat` with wrong token | `401` |
| 4 | Ask something needing a list ("name 5 planets") | one sentence, no `*` or `-` |
| 5 | `npm run logs` while you test | each request shows, with CPU time well under 10 ms |

All five pass → Phase 1 done, move to Phase 2 (S3 calls `/chat`).

## Errors you might see

| Response | Meaning | Fix |
|---|---|---|
| `500 server not configured` | A secret is missing | Re-run the `wrangler secret put` commands |
| `401` | Token mismatch | Same token in `.dev.vars` / secret and in your request |
| `502 gemini 404 ...model...` | Model name not available to your key | Change `GEMINI_MODEL` in `wrangler.toml` to one listed in AI Studio, redeploy |
| `502 gemini 400/403 ...key...` | Bad or restricted key | New key from AI Studio |
| `429` | Free-tier rate limit | Wait a minute; the S3 will play its offline line for this |
| `504` | Gemini took >15 s | Rare; just retry |

## Changing behaviour without reflashing

- **Model:** edit `GEMINI_MODEL` in `wrangler.toml` → `npm run deploy`
- **Personality / length:** edit `SYSTEM_PROMPT` in `src/index.js` → `npm run deploy`
