import express from 'express';
import http from 'http';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import JSZip from 'jszip';
import { GoogleGenAI, Type, ThinkingLevel } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build'
    }
  }
});

// Cache recent AI card detections to conserve API quota across preset switches / re-snaps
const aiDetectionCache = new Map<string, any>();
const modelQuotaCooldownUntil = new Map<string, number>();

async function startServer() {
  const app = express();
  const httpServer = http.createServer(app);

  // Allow up to 100MB for PDF / high-res scan uploads
  app.use(express.json({ limit: '100mb' }));
  app.use(express.urlencoded({ extended: true, limit: '100mb' }));

  // In-memory customer folders storage for shopkeeper print station
  let sharedFolders: any[] = [];

  // API Routes
  app.get('/api/folders', (_req, res) => {
    res.json(sharedFolders);
  });

  app.post('/api/folders', (req, res) => {
    try {
      const folder = req.body;
      if (!folder || !folder.id) {
        return res.status(400).json({ error: 'Invalid folder data' });
      }
      // Add to front of list
      sharedFolders.unshift(folder);
      // Keep maximum 100 recent folders
      if (sharedFolders.length > 100) {
        sharedFolders = sharedFolders.slice(0, 100);
      }
      res.json({ success: true, folder });
    } catch (err: any) {
      res.status(500).json({ error: err.message });
    }
  });

  app.patch('/api/folders/:id', (req, res) => {
    const { id } = req.params;
    const { status } = req.body;
    const item = sharedFolders.find((f) => f.id === id);
    if (item) {
      item.status = status;
    }
    res.json({ success: true });
  });

  app.delete('/api/folders/:id', (req, res) => {
    const { id } = req.params;
    sharedFolders = sharedFolders.filter((f) => f.id !== id);
    res.json({ success: true });
  });

  app.delete('/api/folders', (_req, res) => {
    sharedFolders = [];
    res.json({ success: true });
  });

  // AI-Powered ID Card Auto-Crop Bounding Box Detection (Gemini Vision)
  app.post('/api/ai/detect-aadhaar-cards', async (req, res) => {
    try {
      const {
        imageBase64,
        mimeType = 'image/jpeg',
        cardTypeHint = 'aadhaar',
        pageRole = 'both'
      } = req.body || {};
      if (!imageBase64 || typeof imageBase64 !== 'string') {
        return res.status(400).json({ error: 'imageBase64 is required' });
      }
      if (!process.env.GEMINI_API_KEY) {
        return res.status(500).json({ error: 'GEMINI_API_KEY is not configured on the server.' });
      }

      const cleanBase64 = imageBase64.includes(',') ? imageBase64.split(',')[1] : imageBase64;
      const cacheKey = crypto
        .createHash('sha256')
        .update(`${cardTypeHint}|${pageRole}|${cleanBase64}`)
        .digest('hex');

      const cached = aiDetectionCache.get(cacheKey);
      if (cached) {
        return res.json(cached);
      }

      const boxSchema = {
        type: Type.OBJECT,
        properties: {
          ymin: { type: Type.INTEGER, description: 'Top edge (0-1000 normalized height)' },
          xmin: { type: Type.INTEGER, description: 'Left edge (0-1000 normalized width)' },
          ymax: { type: Type.INTEGER, description: 'Bottom edge (0-1000 normalized height)' },
          xmax: { type: Type.INTEGER, description: 'Right edge (0-1000 normalized width)' }
        },
        required: ['ymin', 'xmin', 'ymax', 'xmax']
      };

      const roleRule =
        pageRole === 'back'
          ? `\nIMPORTANT: Inspect this page for the BACK ID card side. If a BACK card appears, put its tight bounding box in "back" and any QR code inside it in "backQr", set "hasBack": true, "detected": true, and set "front" and "frontQr" to {ymin:0, xmin:0, ymax:0, xmax:0}. If this page contains ONLY printed instructions/text/tables and NO ID card, set "detected": false, "hasBack": false, and all boxes to {ymin:0, xmin:0, ymax:0, xmax:0}.`
          : pageRole === 'front'
          ? `\nIMPORTANT: Inspect this page for the FRONT ID card side. If a FRONT card appears, put its tight bounding box in "front" and any QR code inside it in "frontQr", set "hasBack": false, "detected": true, and set "back" and "backQr" to {ymin:0, xmin:0, ymax:0, xmax:0}. If no ID card appears, set "detected": false.`
          : `\nIf BOTH Front and Back cards appear on this page, populate both "front" and "back" (plus "frontQr"/"backQr" where visible) and set "hasBack": true. If ONLY ONE card appears on this page (even if there is a wide text table/instructions box elsewhere on the page), put ONLY that card in "front" (with "frontQr" if visible), set "hasBack": false, and set "back" & "backQr" to {ymin:0, xmin:0, ymax:0, xmax:0}.`;

      const buildHintInstruction = (hint: string): string => {
        if (hint === 'aadhaar') {
          return `This is an Indian UIDAI e-Aadhaar PDF or scan page.
Ignore the upper half letter text, enrolment slip, and horizontal dashed scissor cut-line + scissors icon.
Locate ONLY the printable CR80 ID cards (usually side-by-side at the bottom, or stacked):
- "front": the COMPLETE Aadhaar Front card rectangle from just below the top dashed scissor line down to the bottom black border line. It contains vertical "Aadhaar no. issued:" along the left edge, top Ashoka Lion Emblem, Saffron/Green brushes with "ভারত সরকার" / "भारत सरकार" / "Government of India" + right Aadhaar Sun logo, resident portrait photo, name/DOB/gender, red disclaimer box, 12-digit Aadhaar number, bottom red divider line, AND the bottom tagline ("আমার আধার, আমার পরিচয়" / "Mera Aadhaar, Meri Pehchan" / regional tagline). Leave 3-4% breathing space ABOVE the top "ভারত সরকার" / Lion Emblem / Aadhaar Sun logo, leave full left margin for "Aadhaar no. issued:", and include the ENTIRE bottom tagline — NEVER clip the top header, Emblem top, or bottom text!
- "back": the COMPLETE Aadhaar Back card rectangle from just below the top dashed scissor line down to the bottom black border line. It contains vertical "Details as on: DD/MM/YYYY" along the left edge, top Ashoka Lion Emblem, Saffron/Green brushes with "ভারতীয় বিশিষ্ট পরিচয় প্রাধিকরণ" / "Unique Identification Authority of India" + right Aadhaar Sun logo, Address block, the LARGE square QR code on the right, 12-digit Aadhaar number/VID, bottom red divider line, AND the bottom footer ("1947 | help@uidai.gov.in | www.uidai.gov.in"). Leave 3-4% breathing space ABOVE the top header / Emblem / Sun logo, include full left "Details as on:", full right "www.uidai.gov.in", and full bottom footer — NEVER crop any edge text!
- "frontQr" / "backQr": tight square box around any QR code on Front or Back ({0,0,0,0} if none).`;
        }
        if (hint === 'pan_card') {
          return `This is an Indian Income Tax / NSDL / Protean / UTIITSL e-PAN PDF or scan/photo.
Ignore the A4 letter text, receipt details, instructions, and horizontal dashed scissor cut-line.
Locate the printable CR80 PAN card(s):
- "front": The blue/cyan PAN Front card with "INCOME TAX DEPARTMENT" and "GOVT. OF INDIA", holder's photo, signature, 10-character PAN number, Name, Father's Name, DOB, and QR code (if present). Crop tightly to the exact outer border/edges of the card rectangle.
- "back": The PAN Back card with large QR code (if present), Income Tax return address, helpline instructions, or outer cut border.
- "frontQr" / "backQr": tight square box around the QR code inside Front or Back card ({0,0,0,0} if none).`;
        }
        if (hint === 'voter_epic') {
          return `This is an Indian Election Commission (ECI / NVSP / voters.eci.gov.in) e-EPIC Voter ID PDF page, scan, or camera photo.
CRITICAL LAYOUT AWARENESS FOR INDIAN VOTER ID DOCUMENTS:
1. 2023-2026 ECI DIGITAL e-EPIC PDFs: The Front & Back Voter ID cards sit SIDE-BY-SIDE IN THE TOP / UPPER HALF of the page (roughly ymin 50..360) separated by a vertical dashed/solid center divider line around xmin=500. Below the cards there is a LARGE wide box/table (80-95% page width) containing "Electoral Photo Identity Card Details", "Polling Station", "Part Number", "Serial No.", and "Important Instructions / Note".
2. STRICT RULE — NEVER CROP THE POLLING STATION OR INSTRUCTIONS TABLE & NEVER MERGE FRONT+BACK: Never treat that wide details/polling-station/instructions box as "back" or "front"! And NEVER combine side-by-side Front and Back into a single wide box in "front" — ALWAYS separate "front" (xmin < 500) and "back" (xmin >= 500) with "hasBack": true! If only the Front card + a wide details/instructions table appear on this page, set "hasBack": false and "back": {ymin:0, xmin:0, ymax:0, xmax:0}.
3. OTHER VOTER LAYOUTS: Older NVSP PDFs place Front & Back side-by-side in the lower half; mobile photos or scans may stack Front on top and Back underneath vertically, show old vertical/portrait laminated cards, or show only a single card side.
4. TIGHT INNER CARD CROP & VOTER BACK RULE:
   - Crop STRICTLY INSIDE the outer dashed scissor cut-line border and vertical center cut line — NEVER include outer white gaps around dashed lines or the dashed scissor lines themselves!
   - VOTER BACK WHITE BACKGROUND RULE: Voter Back often has a mostly white interior background (top-left QR code, Address, Constituency, Date, ERO signature) with empty white space on its right or bottom-right inside the card. Keep the "back" card box at the EXACT SAME physical CR80 width, height, and row/column alignment as the "front" card box (do NOT shrink "back" tightly around only its text/QR, and do NOT shift it sideways into the center divider or outer dashed line!).

Locate ONLY the actual printable Voter ID card faces:
- "front": Voter ID Front card (Ashoka Lion emblem + "ELECTION COMMISSION OF INDIA" / "ভারতের নির্বাচন কমিশন" / "भारत निर्वाचन आयोग", "e-Electoral Photo Identity Card" / "IDENTITY CARD", Elector's photo, EPIC number e.g. 3 letters + 7 digits, Name, Father's/Husband's/Mother's Name, Gender, DOB/Age).
- "back": Voter ID Back card (square QR code usually top-left or barcode on old cards, Address, Assembly/Parliamentary Constituency, ERO signature/seal, Download/Issue Date) keeping full matching CR80 card bounds.
- "frontQr" / "backQr": complete tight square box covering 100% of any QR code (all 3 corner finder squares + timing pattern) inside Front or Back card ({ymin:0, xmin:0, ymax:0, xmax:0} if none).`;
        }
        if (hint === 'wb_ration') {
          return `This is an Indian West Bengal Digital / e-Ration Card PDF page, scan, or camera photo (Department of Food & Supplies, Govt. of West Bengal — food.wb.gov.in / WBPDS).
CRITICAL LAYOUT & COMPLETE CARD ARTWORK RULES FOR WEST BENGAL RATION CARDS:
1. IGNORE OUTER A4 CHROME: Ignore top web portal headers ("food.wb.gov.in", "DEPARTMENT OF FOOD & SUPPLIES"), wide page-width instructions/entitlement tables ("বিশেষ দ্রষ্টব্য" / "Important Instructions"), blank A4 margins, AND outer dashed/solid scissor cut-line borders around the card.
2. CAPTURE THE COMPLETE CARD FACE (NEVER CLIP TEXT OR QR!):
   - Locate the rectangular printable Ration Card face(s):
     * FRONT contains the 3 vertical pale mint-green columns, top-left Ashoka Lion Emblem ("Government of West Bengal"), full Bengali title "পশ্চিমবঙ্গ সরকার / খাদ্য ও সরবরাহ দপ্তর", "Digital Ration Card No:" (RKSY-I / RKSY-II / PHH / SPHH / AAY), Card Holder Name, Father/Husband Name, DOB, Dealer Name & Address, bottom-left "Not Transferable", bottom-center circular green seal, and bottom-right QR code (or barcode).
     * BACK (if present on this page — either side-by-side next to Front, stacked vertically below Front, or alone on Page 2) has roughly the SAME physical card rectangle size as Front and contains Address / Family Head / FPS / Issue Date / Verification / QR or barcode.
   - Hug tightly just outside the printed artwork (top Lion crown + top matras of "পশ্চিমবঙ্গ সরকার" down through the bottom mint edge & "Not Transferable", left mint stripe through right mint stripe/QR edge) — NEVER clip any top Bengali header vowels, bottom text, or QR code! (The app strips residual scissor lines & adds uniform 4.4%/5.3% white PVC cushion afterwards.)
3. FRONT vs BACK RULES:
   - Side-by-side A4: Front and Back share nearly the exact same vertical row (ymin/ymax) and card size. NEVER merge side-by-side Front and Back into one wide box!
   - Vertically stacked A4: Front is above Back with nearly the exact same horizontal width (xmin/xmax) and height.
   - NEVER treat a wide page-width instructions/disclaimer/beneficiary table outside a card as "back"! If only ONE Ration Card face appears on this page, put it in "front", set "hasBack": false, and set "back" & "backQr" to {ymin:0, xmin:0, ymax:0, xmax:0}.`;
        }
        if (hint === 'ayushman') {
          return `This is an Indian Ayushman Bharat (PM-JAY / NHA / ABHA / State Health Card) PDF page or scan/photo.
IMPORTANT:
- Ayushman PDFs are usually 2-page PDFs where Page 1 is the FRONT card and Page 2 is the BACK card, and the card artwork often fills the ENTIRE PDF page (up to full {ymin:0, xmin:0, ymax:1000, xmax:1000}). If the card fills the whole page, return {ymin:0, xmin:0, ymax:1000, xmax:1000} for the card box! Do NOT fail or omit the box when the card touches page edges.
- MOST CRITICAL TASK: Locate the EXACT tight bounding box around every QR code visible on this page!
  * Look closely in all quadrants (top-right, bottom-right, center-left, bottom-left, top-left) for any black/dark QR square matrix.
  * "frontQr": tight square box around the QR code inside the Front card ({ymin:0, xmin:0, ymax:0, xmax:0} ONLY if there is genuinely no QR on the Front).
  * "backQr": tight square box around the QR code inside the Back card ({ymin:0, xmin:0, ymax:0, xmax:0} ONLY if there is genuinely no QR on the Back).
- "front": Full Front card area (PM-JAY / NHA / ABHA header, beneficiary photo, Name, YOB/Gender, PM-JAY / ABHA ID, QR code).
- "back": Full Back card area (PM-JAY benefits / 5 Lakh free treatment / helpline 14555 / QR code / NHA footer).`;
        }
        return `This is an Indian Government ID card PDF or scan/photo (Voter ID e-EPIC, West Bengal e-Ration Card, PAN, Ayushman PM-JAY, Aadhaar, Driving License, or e-Shram).
CRITICAL LAYOUT RULES:
- Identify the card type cleanly in "detectedType" as one of: "wb_ration" (West Bengal Ration Card / খাদ্য ও সরবরাহ দপ্তর / RKSY / PHH / SPHH / AAY / 3 vertical mint-green stripes), "aadhaar" (UIDAI Aadhaar), "pan_card" (Income Tax PAN), "voter_epic" (ECI Voter e-EPIC), "ayushman" (PM-JAY / ABHA), or "other".
- SIDE-BY-SIDE RULE (CRITICAL): On ECI e-EPIC Voter IDs, WB e-Ration Cards, UIDAI Aadhaar, and PAN PDFs, Front (left, xmin < 500) and Back (right, xmin >= 500) cards often sit side-by-side separated by a vertical dashed/solid center line inside a shared dashed scissor box. NEVER merge two side-by-side cards into one double-wide rectangle in "front"! Separate left card into "front" and right card into "back" with "hasBack": true, matching physical card sizes cleanly even when Voter/Ration Back has a mostly white background.
- On ECI e-EPIC Voter IDs, WB e-Ration Cards, and Ayushman PDFs, the printable ID card(s) are often in the TOP or UPPER/MIDDLE section (above Polling Station/Elector Details/Instructions tables). On UIDAI Aadhaar and NSDL/UTI PAN PDFs, the printable cards are usually at the BOTTOM HALF (below letter text and cut lines). Cards may also be stacked vertically, portrait/vertical (older Voter ID), or single-sided.
- NEVER crop A4 letter text, Polling Station / Part No / Serial tables, Important Instructions boxes, portal headers, blank white padding, dashed scissor cut-lines, or center vertical fold/cut lines!
- Detect ONLY the tight bounding box of the actual printed Front ID card ("front"), Back ID card ("back" — must be an actual ID card face, NOT a wide instructions/details table!), and complete square QR codes ("frontQr", "backQr").`;
      };

      const hintInstruction = buildHintInstruction(String(cardTypeHint || 'auto')) + roleRule;
      const now = Date.now();
      const candidateModels = ['gemini-3.1-flash-lite', 'gemini-4-hydrogen'].filter(
        (m) => (modelQuotaCooldownUntil.get(m) || 0) <= now
      );
      const modelsToTry = candidateModels.length > 0 ? candidateModels : ['gemini-3.1-flash-lite'];

      let lastErr: any = null;
      let hitQuota = false;

      for (const modelName of modelsToTry) {
        let modelTimer: ReturnType<typeof setTimeout> | null = null;
        try {
          const timeoutMs = modelName === 'gemini-4-hydrogen' ? 16000 : 11000;
          const response = await Promise.race([
            ai.models.generateContent({
              model: modelName,
              contents: {
                parts: [
                  {
                    inlineData: {
                      mimeType,
                      data: cleanBase64
                    }
                  },
                  {
                    text: `${hintInstruction}\nReturn exact tight bounding boxes in [0..1000] scale where ymin/ymax are vertical (0=top, 1000=bottom) and xmin/xmax are horizontal (0=left, 1000=right). Also set "detectedType" accurately. Set "detected": true if the requested card side is clearly found.`
                  }
                ]
              },
              config: {
                ...(modelName === 'gemini-4-hydrogen'
                  ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
                  : {}),
                responseMimeType: 'application/json',
                responseSchema: {
                  type: Type.OBJECT,
                  properties: {
                    detected: { type: Type.BOOLEAN },
                    hasBack: { type: Type.BOOLEAN },
                    detectedType: {
                      type: Type.STRING,
                      enum: ['aadhaar', 'pan_card', 'voter_epic', 'wb_ration', 'ayushman', 'other']
                    },
                    front: boxSchema,
                    back: boxSchema,
                    frontQr: boxSchema,
                    backQr: boxSchema
                  },
                  required: ['detected', 'hasBack', 'detectedType', 'front', 'back', 'frontQr', 'backQr']
                }
              }
            }),
            new Promise<never>((_, reject) => {
              modelTimer = setTimeout(
                () => reject(new Error(`Model ${modelName} timed out after ${Math.round(timeoutMs / 1000)}s`)),
                timeoutMs
              );
            })
          ]);
          if (modelTimer) clearTimeout(modelTimer);

          const rawText = (response.text || '{}').trim();
          const parsed = JSON.parse(rawText);
          if (parsed?.detected) {
            if (aiDetectionCache.size >= 60) {
              const oldestKey = aiDetectionCache.keys().next().value;
              if (oldestKey) aiDetectionCache.delete(oldestKey);
            }
            aiDetectionCache.set(cacheKey, parsed);
          }
          return res.json(parsed);
        } catch (modelErr: any) {
          if (modelTimer) clearTimeout(modelTimer);
          lastErr = modelErr;
          const status = modelErr?.status || modelErr?.code;
          const msg = String(modelErr?.message || '');
          const isRateLimit =
            status === 429 || msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('quota');
          if (isRateLimit) {
            hitQuota = true;
            modelQuotaCooldownUntil.set(modelName, Date.now() + 2 * 60 * 1000);
            console.info(`Gemini model ${modelName} quota reached; using fallback model.`);
            continue;
          }
          modelQuotaCooldownUntil.set(modelName, Date.now() + 15 * 1000);
          console.info(`Gemini model ${modelName} fallback (${msg})`);
          continue;
        }
      }

      if (hitQuota) {
        return res.status(200).json({
          detected: false,
          quotaExceeded: true,
          message:
            'Free AI daily quota reached. Using local Smart Crop & QR detection (select a billing-enabled key in Settings > Secrets to raise quota).'
        });
      }

      return res.status(200).json({
        detected: false,
        fallback: true,
        message: lastErr?.message || 'Switched to instant local Smart Crop'
      });
    } catch (err: any) {
      const msg = String(err?.message || 'AI card detection failed');
      console.info('AI card detection fallback:', msg);
      return res.status(200).json({
        detected: false,
        fallback: true,
        message: msg
      });
    }
  });

  // AI-Powered Passport Photo Face, Lighting, Alignment & Prompt Background Analysis
  app.post('/api/ai/passport-analyze', async (req, res) => {
    try {
      const { imageBase64, mimeType = 'image/jpeg', promptText = '' } = req.body || {};
      if (!imageBase64 || typeof imageBase64 !== 'string') {
        return res.status(400).json({ error: 'imageBase64 is required' });
      }
      if (!process.env.GEMINI_API_KEY) {
        return res.status(200).json({
          analyzed: false,
          fallback: true,
          message: 'Using local Studio Engine (server API key not configured).'
        });
      }

      const cleanBase64 = imageBase64.includes(',') ? imageBase64.split(',')[1] : imageBase64;
      const cacheKey = crypto
        .createHash('sha256')
        .update(`passport-v4|${promptText.slice(0, 500)}|${cleanBase64.slice(0, 4096)}|${cleanBase64.length}`)
        .digest('hex');

      const cached = aiDetectionCache.get(cacheKey);
      if (cached) {
        return res.json(cached);
      }

      const pointSchema = {
        type: Type.OBJECT,
        properties: {
          x: { type: Type.INTEGER, description: 'Horizontal position (0-1000)' },
          y: { type: Type.INTEGER, description: 'Vertical position (0-1000)' }
        },
        required: ['x', 'y']
      };

      const boxSchema = {
        type: Type.OBJECT,
        properties: {
          ymin: { type: Type.INTEGER, description: 'Top edge (0-1000)' },
          xmin: { type: Type.INTEGER, description: 'Left edge (0-1000)' },
          ymax: { type: Type.INTEGER, description: 'Bottom edge (0-1000)' },
          xmax: { type: Type.INTEGER, description: 'Right edge (0-1000)' }
        },
        required: ['ymin', 'xmin', 'ymax', 'xmax']
      };

      const now = Date.now();
      const candidateModels = ['gemini-3.1-flash-lite', 'gemini-4-hydrogen'].filter(
        (m) => (modelQuotaCooldownUntil.get(m) || 0) <= now
      );
      const modelsToTry = candidateModels.length > 0 ? candidateModels : ['gemini-3.1-flash-lite'];

      let hitQuota = false;
      let lastErr: any = null;

      for (const modelName of modelsToTry) {
        let modelTimer: ReturnType<typeof setTimeout> | null = null;
        try {
          const timeoutMs = modelName === 'gemini-4-hydrogen' ? 14000 : 9000;
          const response = await Promise.race([
            ai.models.generateContent({
              model: modelName,
              contents: {
                parts: [
                  {
                    inlineData: {
                      mimeType,
                      data: cleanBase64
                    }
                  },
                  {
                    text: `You are a professional passport photo studio vision analyzer (Free Gemini Vision).
Analyze this portrait photo accurately for face location, full hair/head crown bounds, eye centers, shoulder bounds, head tilt, lighting balance, and studio retouching.
Studio Prompt context:
"""
${promptText || 'Clean natural studio retouching.'}
"""

Return exact normalized coordinates [0..1000] where 0,0 is top-left and 1000,1000 is bottom-right:
1. "faceBox": box enclosing the FULL head from the VERY TOP OF THE HAIR/CROWN/HIJAB down to the bottom of the chin, and horizontally across both ears/hair sides.
2. "personBox": full bounding box of the visible person (very top of hair down to lowest shirt edge, full outer shoulders/arms).
3. "eyeLeft" & "eyeRight": exact center of wearer's left eye and right eye.
4. "hairTopY": Y coordinate [0..1000] of the very highest hair strand / turban / hijab edge (never cut into dark hair!).
5. "chinY": Y coordinate [0..1000] of the bottom edge of the chin.
6. "shoulderY": Y coordinate [0..1000] where collar/shoulders slope outward.
7. "shoulderLeftX" & "shoulderRightX": X coordinates [0..1000] of OUTERMOST left and right shoulder/clothing edges.
8. "bgComplexity": "uniform" | "gradient" | "cluttered".
9. "headTiltDeg": rotation (-15 to +15) in degrees to level the eyes horizontally.
10. "brightnessDelta": integer (-25 to +35) exposure adjustment for balanced natural skin tones.
11. "contrastBoost": number (0.95 to 1.25).
12. "skinSmoothRecommended": integer (20 to 65).
13. "shadowLiftRecommended": integer (20 to 65).
14. "harshShadowSide": "none" | "left" | "right" | "bottom".
15. "targetBgHex": exact 6-digit hex color mentioned in prompt (default "#1D6FE8").
16. "promptSummary": short 1-line summary (e.g. "Face & Hair Locked • Studio Light • Clean Framing").`
                  }
                ]
              },
              config: {
                ...(modelName === 'gemini-4-hydrogen'
                  ? { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } }
                  : {}),
                responseMimeType: 'application/json',
                responseSchema: {
                  type: Type.OBJECT,
                  properties: {
                    analyzed: { type: Type.BOOLEAN },
                    faceBox: boxSchema,
                    personBox: boxSchema,
                    eyeLeft: pointSchema,
                    eyeRight: pointSchema,
                    hairTopY: { type: Type.INTEGER },
                    chinY: { type: Type.INTEGER },
                    shoulderY: { type: Type.INTEGER },
                    shoulderLeftX: { type: Type.INTEGER },
                    shoulderRightX: { type: Type.INTEGER },
                    bgComplexity: { type: Type.STRING },
                    headTiltDeg: { type: Type.NUMBER },
                    brightnessDelta: { type: Type.INTEGER },
                    contrastBoost: { type: Type.NUMBER },
                    skinSmoothRecommended: { type: Type.INTEGER },
                    shadowLiftRecommended: { type: Type.INTEGER },
                    harshShadowSide: { type: Type.STRING },
                    targetBgHex: { type: Type.STRING },
                    promptSummary: { type: Type.STRING }
                  },
                  required: [
                    'analyzed',
                    'faceBox',
                    'personBox',
                    'eyeLeft',
                    'eyeRight',
                    'hairTopY',
                    'chinY',
                    'shoulderY',
                    'shoulderLeftX',
                    'shoulderRightX',
                    'bgComplexity',
                    'headTiltDeg',
                    'brightnessDelta',
                    'contrastBoost',
                    'skinSmoothRecommended',
                    'shadowLiftRecommended',
                    'harshShadowSide',
                    'targetBgHex',
                    'promptSummary'
                  ]
                }
              }
            }),
            new Promise<never>((_, reject) => {
              modelTimer = setTimeout(
                () => reject(new Error(`Model ${modelName} timed out after ${Math.round(timeoutMs / 1000)}s`)),
                timeoutMs
              );
            })
          ]);
          if (modelTimer) clearTimeout(modelTimer);

          const rawText = (response.text || '{}').trim();
          const parsed = JSON.parse(rawText);
          if (aiDetectionCache.size >= 60) {
            const oldestKey = aiDetectionCache.keys().next().value;
            if (oldestKey) aiDetectionCache.delete(oldestKey);
          }
          aiDetectionCache.set(cacheKey, parsed);
          return res.json(parsed);
        } catch (modelErr: any) {
          if (modelTimer) clearTimeout(modelTimer);
          lastErr = modelErr;
          const status = modelErr?.status || modelErr?.code;
          const msg = String(modelErr?.message || '');
          const isRateLimit =
            status === 429 || msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED') || msg.includes('quota');
          if (isRateLimit) {
            hitQuota = true;
            modelQuotaCooldownUntil.set(modelName, Date.now() + 2 * 60 * 1000);
            continue;
          }
          modelQuotaCooldownUntil.set(modelName, Date.now() + 15 * 1000);
          continue;
        }
      }

      if (hitQuota) {
        return res.status(200).json({
          analyzed: false,
          quotaExceeded: true,
          message: 'Free AI quota reached; using built-in local Passport Studio engine.'
        });
      }

      return res.status(200).json({
        analyzed: false,
        fallback: true,
        message: lastErr?.message || 'Using built-in local Passport Studio engine.'
      });
    } catch (err: any) {
      const msg = String(err?.message || 'Using built-in local Passport Studio engine.');
      console.info('Passport AI analyze fallback:', msg);
      return res.status(200).json({
        analyzed: false,
        fallback: true,
        message: msg
      });
    }
  });

  // Download packaged Android App bundle from /Download/App folder as SakibStudio_Android_App.zip
  app.get('/api/download-app', async (req, res) => {
    try {
      const appDir = path.resolve(__dirname, 'Download', 'App');
      if (!fs.existsSync(appDir)) {
        return res.status(404).json({ error: 'Download/App folder not found' });
      }

      const forwardedProto = (req.headers['x-forwarded-proto'] as string)?.split(',')[0]?.trim() || req.protocol || 'https';
      const rawHost = (req.headers['x-forwarded-host'] as string)?.split(',')[0]?.trim() || req.get('host') || '';
      const publicHost = rawHost.startsWith('ais-dev-') ? rawHost.replace(/^ais-dev-/, 'ais-pre-') : rawHost;
      const activeUrl =
        publicHost && !publicHost.startsWith('localhost') && !publicHost.startsWith('127.')
          ? `${forwardedProto}://${publicHost}/`
          : 'https://ais-pre-ornr6xbauv76a3r6y2t4zb-252294331872.asia-east1.run.app/';

      const zip = new JSZip();
      const rootFolder = zip.folder('Download')!.folder('App')!;

      const addDirToZip = (dirPath: string, zipFolder: JSZip) => {
        const entries = fs.readdirSync(dirPath, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dirPath, entry.name);
          if (entry.isDirectory()) {
            const subFolder = zipFolder.folder(entry.name)!;
            addDirToZip(fullPath, subFolder);
          } else {
            const ext = path.extname(entry.name).toLowerCase();
            if (['.txt', '.kt', '.xml', '.kts', '.html'].includes(ext)) {
              const raw = fs.readFileSync(fullPath, 'utf8');
              const updated = raw.replace(
                /https:\/\/ais-pre-ornr6xbauv76a3r6y2t4zb-252294331872\.asia-east1\.run\.app\//g,
                activeUrl
              );
              zipFolder.file(entry.name, updated);
            } else {
              zipFolder.file(entry.name, fs.readFileSync(fullPath));
            }
          }
        }
      };

      addDirToZip(appDir, rootFolder);

      const zipBuffer = await zip.generateAsync({
        type: 'nodebuffer',
        compression: 'DEFLATE',
        compressionOptions: { level: 6 }
      });

      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', 'attachment; filename="SakibStudio_Android_App.zip"');
      res.setHeader('Cache-Control', 'no-store');
      res.setHeader('Content-Length', String(zipBuffer.length));
      return res.send(zipBuffer);
    } catch (err: any) {
      return res.status(500).json({ error: err?.message || 'Failed to package App download' });
    }
  });

  // Health check endpoint
  app.get('/api/health', (_req, res) => {
    res.json({ status: 'ok', foldersCount: sharedFolders.length });
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    // HMR is disabled in AI Studio; serve a WebSocket-free /@vite/client shim before Vite
    // so CSS injection & module stubs work cleanly without failing WebSocket handshakes behind proxies.
    app.get('/@vite/client', (_req, res) => {
      res.setHeader('Content-Type', 'application/javascript; charset=utf-8');
      res.setHeader('Cache-Control', 'no-store');
      res.send(`
const sheetsMap = new Map();
const linkSheetsMap = new Map();
if ("document" in globalThis) {
  document.querySelectorAll("style[data-vite-dev-id]").forEach((el) => {
    sheetsMap.set(el.getAttribute("data-vite-dev-id"), el);
  });
  document.querySelectorAll('link[rel="stylesheet"][data-vite-dev-id]').forEach((el) => {
    linkSheetsMap.set(el.getAttribute("data-vite-dev-id"), el);
  });
}
let lastInsertedStyle;
export function updateStyle(id, content) {
  if (linkSheetsMap.has(id)) return;
  let style = sheetsMap.get(id);
  if (!style) {
    style = document.createElement("style");
    style.setAttribute("type", "text/css");
    style.setAttribute("data-vite-dev-id", id);
    style.textContent = content;
    if (!lastInsertedStyle) {
      document.head.appendChild(style);
      setTimeout(() => { lastInsertedStyle = undefined; }, 0);
    } else {
      lastInsertedStyle.insertAdjacentElement("afterend", style);
    }
    lastInsertedStyle = style;
  } else if (style.textContent !== content) {
    style.textContent = content;
  }
  sheetsMap.set(id, style);
}
export function removeStyle(id) {
  if (linkSheetsMap.has(id)) {
    document.querySelectorAll('link[rel="stylesheet"][data-vite-dev-id="' + CSS.escape(id) + '"]').forEach((el) => el.remove());
    linkSheetsMap.delete(id);
  }
  const style = sheetsMap.get(id);
  if (style) {
    style.remove();
    sheetsMap.delete(id);
  }
}
export function createHotContext() {
  return {
    data: {},
    accept() {},
    acceptExports() {},
    dispose() {},
    prune() {},
    decline() {},
    invalidate() {},
    on() {},
    off() {},
    send() {}
  };
}
export function injectQuery(url, queryToInject) {
  if (url[0] !== "." && url[0] !== "/") return url;
  const pathname = url.replace(/[?#].*$/, "");
  const { search, hash } = new URL(url, "http://vite.dev");
  return pathname + "?" + queryToInject + (search ? "&" + search.slice(1) : "") + (hash || "");
}
export class ErrorOverlay {
  constructor() {}
  close() {}
}
`);
    });

    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        hmr: false,
        ws: false,
        watch: null
      },
      appType: 'spa'
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  const PORT = process.env.PORT || 3000;
  httpServer.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`Jasmin Cyber Cafe Print Server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer();
