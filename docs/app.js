const FIGURE_CAPTIONS = {
  Fig1: "Figure 1. Yangtze River Basin subregions (Upper, Middle, and Lower catchments), seven fractional HydroBASINS Level-6 tributaries, hydrometric infrastructure (Three Gorges Dam at Yichang; Datong mainstem gauge), and 1961–2025 monthly precipitation medians & IQRs paired with median temperature highlighting May–Sep flood-season onset prevalence (41.0%, n = 16).",
  Fig2: "Figure 2. Comparison of basin-mean F_w versus synchronous peak tributary F_w across 39 events coded by severity and season. Mild whole-basin wet recoveries (F_w < 1.0) exhibit a median tributary SPI amplification of 1.84x vs. 1.24x for F_w >= 1.0 (MW p = 1.8e-4), uncoupled level r_s = +0.689 (p = 1.0e-6), and a 3.29x median physical rainfall excess ratio (14.23 km^3).",
  Fig3: "Figure 3. (a) Complete 1961–2025 chronology of basin-wide SPI-1 transitions showing 11-yr centered transition rates alongside preceding drought (-S_d) and subsequent wet (+F_w) severities (maximum jump J_w = 3.02 in 1989-01). (b) Monthly transition onset distribution across Flood (41.0%), Winter (33.3%), and Shoulder (25.6%).",
  Fig4: "Figure 4. (a) Absence of statistically detectable S_d vs. F_w scaling at whole-basin scale (n = 39, tau = -0.139, p = 0.213; 80%-power MDE |tau| = 0.313, |r| = 0.436; Gamma best marginals; Q50/Q90 slopes non-significant). (b) Grid-cell Kendall tau across 748 CRU cells (median +0.008; 47/748 nominal p < 0.05 vs. seasonal AR(1) field null P95 = 7.5%, field p = 0.183; 0/748 BH-FDR q < 0.05).",
  Fig5: "Figure 5. (a) Sensitivity matrix across 36 definition schemes and forest-plot comparison across SPI-1, SPI-3, Penman–Monteith and Thornthwaite SPEI-1/3, and 1-month bridging with per-sample Kendall tau 80%-MDE bands. (b) Datong mainstem wet-spell discharge anomalies stratified by basin F_w, season (filled Flood vs. open Cool), and regulation eras (1961–2024).",
  Fig6: "Figure 6. (a) Detrended 500 hPa geopotential height anomalies (Z_500, stippling p < 0.05), p_s-bounded moisture transport vectors (Q'), and detrended month-matched 5880 gpm WNPSH edges (135.0 deg E climatology vs. 132.5 deg E transitions) for May–Sep flood transitions (n = 16). (b) All-event composite (n = 39). (c–d) Window trajectories [t_w0-3 .. t_w0+1] for DIV_ps and DI500 with separate n = 16 and n = 39 climatological 5–95% bands.",
  ESM_Fig1: "Figure S1. 200 hPa geopotential height anomalies (shading), month-matched May–Sep 25 and 30 m/s upper-tropospheric jet core contours (purple), and horizontal Takaya–Nakamura (2001) wave-activity flux vectors W (|W| >= 0.08 m^2 s^-2 north of 20 deg N; reference 1.0 m^2 s^-2) across East Asia during May–Sep flood transitions (n = 16)."
};

function parseCSVLine(line) {
  const out = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out;
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = parseCSVLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cols = parseCSVLine(line);
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = cols[i] ?? "";
    });
    return obj;
  });
}

function seasonBucket(monthNum) {
  const m = Number(monthNum);
  if (m >= 5 && m <= 9) return "flood";
  if (m === 12 || m === 1 || m === 2) return "winter";
  return "shoulder";
}

function seasonBadge(bucket) {
  if (bucket === "flood") return '<span class="tag tag-flood">Flood (May–Sep)</span>';
  if (bucket === "winter") return '<span class="tag tag-winter">Winter (Dec–Feb)</span>';
  return '<span class="tag tag-shoulder">Shoulder</span>';
}

async function initPortal() {
  const [catText, tribText, subText, copText, circText] = await Promise.all([
    fetch("assets/data/event_catalog.csv").then((r) => r.text()),
    fetch("assets/data/tributary_amplification.csv").then((r) => r.text()),
    fetch("assets/data/subregion_summary.csv").then((r) => r.text()),
    fetch("assets/data/copula_fits.csv").then((r) => r.text()),
    fetch("assets/data/circulation_summary.csv").then((r) => r.text()),
  ]);

  const events = parseCSV(catText);
  const tribs = parseCSV(tribText);
  const subregions = parseCSV(subText);
  const copulas = parseCSV(copText);
  const circ = parseCSV(circText);

  const tribById = new Map(tribs.map((t) => [String(t.event_id), t]));

  // Populate subregion + copula summary table
  const subTbody = document.querySelector("#subregion-table tbody");
  subTbody.innerHTML = "";
  subregions.forEach((s) => {
    const regCop = copulas.filter((c) => c.region === s.region);
    regCop.sort((a, b) => Number(a.AICc) - Number(b.AICc));
    const best = regCop[0] || { copula: "—", delta_AICc_vs_Indep: "0.00", CvM_p: "" };
    const tr = document.createElement("tr");
    const bestLbl = best.CvM_p ? `${best.copula} (CvM p=${Number(best.CvM_p).toFixed(3)})` : best.copula;
    tr.innerHTML = `
      <td><strong>${s.region}</strong></td>
      <td>${s.n}</td>
      <td>${Number(s.kendall_tau).toFixed(3)} (${Number(s.kendall_p).toFixed(3)})</td>
      <td>${Number(s.spearman_r).toFixed(3)} (${Number(s.spearman_p).toFixed(3)})</td>
      <td>${bestLbl}</td>
      <td>${Number(best.delta_AICc_vs_Indep).toFixed(2)}</td>
    `;
    subTbody.appendChild(tr);
  });

  // Populate primary circulation summary rows
  const circTbody = document.querySelector("#circ-table tbody");
  circTbody.innerHTML = "";
  const showRows = circ.filter(
    (c) => ["DIV_ps", "DI500", "DI200", "EAP850"].includes(c.index) &&
           ["All (1961-2025)", "Flood (May-Sep)", "Winter (Dec-Feb)"].includes(c.stratum)
  );
  showRows.forEach((c) => {
    const tr = document.createElement("tr");
    const qStr = c.q_FDR ? Number(c.q_FDR).toFixed(4) : "—";
    tr.innerHTML = `
      <td><strong>${c.index}</strong></td>
      <td>${c.stratum}</td>
      <td>${c.n}</td>
      <td>${Number(c.mean_diff) >= 0 ? "+" : ""}${Number(c.mean_diff).toFixed(3)}&sigma;</td>
      <td>${qStr}</td>
    `;
    circTbody.appendChild(tr);
  });

  // Catalog filtering
  const catTbody = document.querySelector("#catalog-table tbody");
  const searchEl = document.getElementById("search-input");
  const seasonEl = document.getElementById("season-filter");
  const sevEl = document.getElementById("severity-filter");
  const countEl = document.getElementById("catalog-count");

  function renderCatalog() {
    const q = searchEl.value.trim().toLowerCase();
    const sFilt = seasonEl.value;
    const fFilt = sevEl.value;

    const filtered = events.filter((ev) => {
      const trInfo = tribById.get(String(ev.event_id)) || {};
      const sB = seasonBucket(ev.trans_month);
      if (sFilt !== "all" && sB !== sFilt) return false;
      const fw = Number(ev.F_w);
      if (fFilt === "mild" && fw >= 1.0) return false;
      if (fFilt === "modsev" && fw < 1.0) return false;
      if (q) {
        const hay = `${ev.dry_start} ${ev.wet_start} ${ev.year} ${trInfo.peak_tributary || ""}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    countEl.textContent = `Showing ${filtered.length} / ${events.length} events`;
    catTbody.innerHTML = "";
    filtered.forEach((ev) => {
      const trInfo = tribById.get(String(ev.event_id)) || {};
      const sB = seasonBucket(ev.trans_month);
      const qWet = ev.Q_wet_z !== "" && !Number.isNaN(Number(ev.Q_wet_z))
        ? `${Number(ev.Q_wet_z) >= 0 ? "+" : ""}${Number(ev.Q_wet_z).toFixed(2)}&sigma;`
        : "n/a (2025)";
      const exMm = trInfo.basin_precip_ex_mm !== undefined ? `${Number(trInfo.basin_precip_ex_mm) >= 0 ? "+" : ""}${Number(trInfo.basin_precip_ex_mm).toFixed(1)}` : "—";
      const rowEl = document.createElement("tr");
      rowEl.innerHTML = `
        <td>${ev.event_id}</td>
        <td>${ev.dry_start} &rarr; ${ev.dry_end}</td>
        <td>${ev.wet_start} &rarr; ${ev.wet_end}</td>
        <td>${seasonBadge(sB)}</td>
        <td>${Number(ev.S_d).toFixed(2)}</td>
        <td>${Number(ev.F_w).toFixed(2)}</td>
        <td>${trInfo.peak_tributary || "—"} (${trInfo.F_w_peak_trib ? Number(trInfo.F_w_peak_trib).toFixed(2) : "—"})</td>
        <td>${trInfo.spi_amp_ratio ? Number(trInfo.spi_amp_ratio).toFixed(2) + "&times;" : "—"}</td>
        <td>${exMm}</td>
        <td>${qWet}</td>
      `;
      catTbody.appendChild(rowEl);
    });
  }

  searchEl.addEventListener("input", renderCatalog);
  seasonEl.addEventListener("change", renderCatalog);
  sevEl.addEventListener("change", renderCatalog);
  renderCatalog();

  // Figure switcher
  const figImg = document.getElementById("fig-img");
  const figCap = document.getElementById("fig-caption");
  figCap.textContent = FIGURE_CAPTIONS.Fig1;
  document.querySelectorAll(".fig-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".fig-tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const key = btn.dataset.fig;
      figImg.src = `assets/figures/${key}.png`;
      figCap.textContent = FIGURE_CAPTIONS[key] || "";
    });
  });

  // BibTeX copy
  const copyBtn = document.getElementById("copy-bib");
  copyBtn.addEventListener("click", async () => {
    const txt = document.getElementById("bibtex-block").textContent;
    await navigator.clipboard.writeText(txt);
    copyBtn.textContent = "Copied";
    setTimeout(() => { copyBtn.textContent = "Copy BibTeX"; }, 1800);
  });
}

document.addEventListener("DOMContentLoaded", initPortal);
