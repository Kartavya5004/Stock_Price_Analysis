// Dependency-free SVG price chart: close line, two moving averages, hover/touch readout.
const NS = "http://www.w3.org/2000/svg";
const MARGIN = { top: 14, right: 54, bottom: 28, left: 6 };

const svgEl = (name, attrs = {}) => {
  const el = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
};

const utc = (iso) => new Date(`${iso}T00:00:00Z`);
export const fmtDate = (iso) =>
  utc(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });

function niceTicks(min, max, count = 4) {
  const raw = (max - min) / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) ticks.push(v);
  return { ticks, decimals: Math.max(0, -Math.floor(Math.log10(step))) };
}

export function createPriceChart(host) {
  let data = [];
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;

  function clear() {
    host.querySelector("svg")?.remove();
    host.querySelector(".status")?.remove();
    tip.hidden = true;
  }

  function message(text) {
    data = [];
    clear();
    const p = document.createElement("p");
    p.className = "status";
    p.textContent = text;
    host.append(p);
  }

  function draw() {
    if (data.length < 2) return;
    clear();
    const W = Math.max(280, host.clientWidth || 640);
    const H = W < 520 ? 240 : 320;
    const plotW = W - MARGIN.left - MARGIN.right;
    const plotH = H - MARGIN.top - MARGIN.bottom;

    const vals = data.flatMap((p) => [p.close, p.sma10, p.sma30]).filter((v) => v != null);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const pad = (hi - lo || hi * 0.02) * 0.06;
    const yMin = lo - pad;
    const yMax = hi + pad;
    const x = (i) => MARGIN.left + (i / (data.length - 1)) * plotW;
    const y = (v) => MARGIN.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;

    const first = data[0];
    const last = data[data.length - 1];
    const svg = svgEl("svg", {
      viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
      "aria-label": `Closing price from ${fmtDate(first.date)} to ${fmtDate(last.date)}: ${first.close.toFixed(2)} at the start, ${last.close.toFixed(2)} at the end.`,
    });

    // horizontal grid + price labels on the right
    const { ticks, decimals } = niceTicks(yMin, yMax);
    for (const t of ticks) {
      svg.append(svgEl("line", { class: "grid", x1: MARGIN.left, x2: MARGIN.left + plotW, y1: y(t), y2: y(t) }));
      const label = svgEl("text", { class: "tick", x: MARGIN.left + plotW + 8, y: y(t) + 4 });
      label.textContent = t.toLocaleString("en", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
      svg.append(label);
    }

    // month labels, skipping any that would crowd the previous one
    let lastLabelX = -Infinity;
    data.forEach((p, i) => {
      if (i === 0 || utc(p.date).getUTCMonth() === utc(data[i - 1].date).getUTCMonth()) return;
      if (x(i) - lastLabelX < 44 || x(i) > MARGIN.left + plotW - 20) return;
      const d = utc(p.date);
      const isJan = d.getUTCMonth() === 0;
      const label = svgEl("text", { class: "tick", x: x(i), y: H - 8, "text-anchor": "middle" });
      label.textContent = d.toLocaleDateString("en", { month: "short", timeZone: "UTC" }) + (isJan ? ` ${d.getUTCFullYear()}` : "");
      svg.append(label);
      lastLabelX = x(i);
    });

    const path = (key) => {
      let d = "";
      let pen = false;
      data.forEach((p, i) => {
        const v = p[key];
        if (v == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };
    svg.append(svgEl("path", { class: "line-sma30", d: path("sma30") }));
    svg.append(svgEl("path", { class: "line-sma10", d: path("sma10") }));
    svg.append(svgEl("path", { class: "line-close", d: path("close") }));
    svg.append(svgEl("circle", { class: "dot", cx: x(data.length - 1), cy: y(last.close), r: 4.5 }));

    // hover / touch readout
    const cursor = svgEl("line", { class: "cursor", y1: MARGIN.top, y2: MARGIN.top + plotH, visibility: "hidden" });
    const hoverDot = svgEl("circle", { class: "dot", r: 4.5, visibility: "hidden" });
    const hit = svgEl("rect", { x: MARGIN.left, y: MARGIN.top, width: plotW, height: plotH, fill: "transparent" });
    svg.append(cursor, hoverDot, hit);

    const show = (evt) => {
      const box = svg.getBoundingClientRect();
      const px = ((evt.clientX - box.left) / box.width) * W;
      const i = Math.max(0, Math.min(data.length - 1, Math.round(((px - MARGIN.left) / plotW) * (data.length - 1))));
      const p = data[i];
      cursor.setAttribute("x1", x(i)); cursor.setAttribute("x2", x(i)); cursor.setAttribute("visibility", "visible");
      hoverDot.setAttribute("cx", x(i)); hoverDot.setAttribute("cy", y(p.close)); hoverDot.setAttribute("visibility", "visible");

      tip.replaceChildren();
      const head = document.createElement("b");
      head.textContent = p.close.toFixed(2);
      const rows = [fmtDate(p.date)];
      if (p.sma10 != null) rows.push(`10-day avg ${p.sma10.toFixed(2)}`);
      if (p.sma30 != null) rows.push(`30-day avg ${p.sma30.toFixed(2)}`);
      tip.append(head, ...rows.map((t) => Object.assign(document.createElement("span"), { textContent: t })));
      tip.hidden = false;
      const left = (x(i) / W) * host.clientWidth;
      const flip = left > host.clientWidth * 0.6;
      tip.style.left = flip ? "auto" : `${left + 12}px`;
      tip.style.right = flip ? `${host.clientWidth - left + 12}px` : "auto";
    };
    const hide = () => {
      cursor.setAttribute("visibility", "hidden");
      hoverDot.setAttribute("visibility", "hidden");
      tip.hidden = true;
    };
    hit.addEventListener("pointermove", show);
    hit.addEventListener("pointerdown", show);
    hit.addEventListener("pointerleave", hide);

    host.prepend(svg);
    if (!tip.isConnected) host.append(tip);
  }

  let lastWidth = 0;
  new ResizeObserver(() => {
    if (host.clientWidth !== lastWidth) { lastWidth = host.clientWidth; draw(); }
  }).observe(host);

  return {
    set(points) { data = points; draw(); },
    message,
  };
}
