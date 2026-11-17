const FIGURE_CAPTIONS = {
  Fig1: "Figure 1. Yangtze River Basin subregions (Upper, Middle, and Lower catchments), seven primary tributaries, key hydrometric infrastructure (Three Gorges Dam at Yichang; Datong mainstem gauge), and 1961–2025 monthly precipitation regime highlighting May–Sep flood-season onset prevalence (41.0%).",
  Fig2: "Figure 2. Comparison of basin-mean F_w versus peak tributary F_w across 39 events. Mild whole-basin wet recoveries (F_w < 1.0) exhibit a median tributary amplification of 1.82x compared to 1.23x for events with F_w >= 1.0, accompanied by a median 3.26x physical precipitation excess (mm) at the tributary scale.",
  Fig3: "Figure 3. (a) Complete 1961–2025 chronology of basin-wide drought-to-flood abrupt alternations showing preceding drought severity (-S_d) and subsequent wet severity (+F_w) alongside the 2003 Three Gorges impoundment and 2013 cascade commissioning markers. (b) Monthly transition onset distribution.",
  Fig4: "Figure 4. (a) Absence of statistically detectable monotonic scaling between S_d and F_w at whole-basin scale (n = 39, tau = -0.123, p = 0.271; 80%-power MDE |r| = 0.436; copula |DeltaAICc| <= 0.44). (b) Grid-cell Kendall tau across 748 CRU cells (median +0.008; 0/748 significant under BH-FDR q < 0.05).",
  Fig5: "Figure 5. (a) Sensitivity matrix across 36 definition schemes (2/36 nominal p < 0.05), multi-scale indices (SPI-1, SPI-3, SPEI-1, SPEI-3), and 1-month weak-break bridging. (b) Datong mainstem wet-spell discharge anomalies stratified by basin F_w and reservoir eras (1961–2024).",
  Fig6: "Figure 6. (a) Composite 500 hPa geopotential anomalies (shading), surface-bounded moisture transport vectors (Q'), and 5880 gpm contour for May–Sep flood transitions (n = 16) vs. May–Sep climatology. (b) All-season composite (n = 39). (c–d) Window trajectories [t_0-3 .. t_0+1] for DIV_ps* and DI500.",
  ESM_Fig1: "Figure S1. 200 hPa geopotential height anomalies (shading), May–Sep 25 and 30 m/s upper-tropospheric jet core contours (purple), and horizontal Takaya–Nakamura (2001) wave-activity flux vectors W across East Asia during May–Sep flood-season transitions (n = 16)."
};

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
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
    const regCop = copulas.filter((c) => c.region === s.region && c.copula !== "Independence");
    regCop.sort((a, b) => Number(a.AICc) - Number(b.AICc));
    const best = regCop[0] || { copula: "—", delta_AICc_vs_Indep: "0.00" };
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${s.region}</strong></td>
      <td>${s.n}</td>
      <td>${Number(s.kendall_tau).toFixed(3)} (${Number(s.kendall_p).toFixed(3)})</td>
      <td>${Number(s.spearman_r).toFixed(3)} (${Number(s.spearman_p).toFixed(3)})</td>
      <td>${best.copula}</td>
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
