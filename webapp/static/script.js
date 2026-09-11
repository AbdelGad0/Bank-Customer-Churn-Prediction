const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const I18N = {
  en: {
    badge_model: "XGBoost model ✓", badge_thr: "Decision threshold",
    hero_title: "Will your customer stay with the bank?",
    hero_sub: "An ML-powered tool that estimates the <b>probability a customer will churn</b> (stop banking with you), trained on 10,000 customers with a <b>ROC-AUC of 0.87</b>.",
    chip_recall: "Recall 83%", chip_pr: "PR-AUC 0.73", chip_cases: "100K+ samples",
    tab_single: "Single Customer", tab_batch: "Upload CSV / Excel",
    single_title: "Customer Details",
    f_age: "Age", f_country: "Country", f_gender: "Gender",
    f_balance: "Current Balance (€)", f_products: "Number of Products", f_active: "Active Member",
    opt_france: "France", opt_germany: "Germany", opt_spain: "Spain",
    opt_female: "Female", opt_male: "Male", opt_active: "Active", opt_inactive: "Inactive",
    btn_predict: "Predict Now",
    result_title: "Prediction Result",
    gauge_label: "Churn probability",
    risk_high: "⚠ High Churn Risk", risk_low: "✓ Low Churn Risk",
    hint_pre: "Probability above the decision threshold",
    hint_post: "means high churn risk",
    btn_new: "New Prediction",
    batch_title: "Batch Prediction — Upload a Data File",
    dz_main: "Drag & drop your file here, or <span class='dz-link'>click to browse</span>",
    dz_hint: "Accepted formats: <b>.csv · .xlsx · .xls</b>",
    tpl_label: "New here?", tpl_link: "Download a sample file",
    progress: "Processing file and predicting...",
    col_conv: "Churn Probability", col_pred: "Prediction", col_risk: "Risk",
    col_country: "Country", col_gender: "Gender", col_age: "Age", col_balance: "Balance",
    col_products: "No. of Products", col_active: "Active Member",
    risk_tag_high: "High Risk", risk_tag_low: "Low",
    pred_yes: "Yes", pred_no: "No",
    sum_total: "Total Customers", sum_high: "High Churn Risk", sum_rate: "Risk Rate",
    sum_low: "Low Risk", sum_thresh: "Decision Threshold", sum_file: "Processed File",
    note_shown: "Showing first {shown} of {total} customers — download the full results CSV.",
    note_all: "{total} customers in total — download the full results CSV.",
    btn_download: "Download Results (CSV)",
    err_ext: "Please upload a file in CSV or Excel format",
    err_unexpected: "Something went wrong",
    footer: "ChurnScope · Bank Customer Churn Prediction — built with Python / Flask / XGBoost",
    switch_lang: "العربية",
  },
  ar: {
    badge_model: "نموذج XGBoost ✓", badge_thr: "عتبة القرار",
    hero_title: "هل سيبقى عميلك مع البنك؟",
    hero_sub: "أداة ذكية تعتمد على تعلّم الآلة لتقدير <b>احتمالية توقف العميل</b> عن التعامل مع البنك، مبنية على بيانات 10,000 عميل بدقة تمييز <b>ROC-AUC 0.87</b>.",
    chip_recall: "Recall 83%", chip_pr: "PR-AUC 0.73", chip_cases: "أكثر من 100 ألف حالة",
    tab_single: "تنبؤ عميل واحد", tab_batch: "رفع ملف CSV / Excel",
    single_title: "بيانات العميل",
    f_age: "العمر", f_country: "الدولة", f_gender: "الجنس",
    f_balance: "الرصيد الحالي (€)", f_products: "عدد المنتجات", f_active: "عضو نشط",
    opt_france: "فرنسا", opt_germany: "ألمانيا", opt_spain: "إسبانيا",
    opt_female: "أنثى", opt_male: "ذكر", opt_active: "نشط", opt_inactive: "غير نشط",
    btn_predict: "تنبؤ الآن",
    result_title: "نتيجة التنبؤ",
    gauge_label: "احتمالية التوقف",
    risk_high: "⚠ خطر توقف مرتفع", risk_low: "✓ احتمال توقف منخفض",
    hint_pre: "الاحتمالية أكبر من عتبة القرار",
    hint_post: "= خطر توقف مرتفع",
    btn_new: "تنبؤ جديد",
    batch_title: "تنبؤ جماعي — رفع ملف بيانات",
    dz_main: "اسحب ملفك وأفلته هنا، أو <span class='dz-link'>اضغط للاختيار</span>",
    dz_hint: "يتقبّل الملف: <b>.csv · .xlsx · .xls</b>",
    tpl_label: "لم تجرّب بعد؟", tpl_link: "تحميل ملف نموذجي",
    progress: "جارٍ معالجة الملف والتنبؤ...",
    col_conv: "احتمالية التوقف", col_pred: "التنبؤ", col_risk: "الخطر",
    col_country: "الدولة", col_gender: "الجنس", col_age: "العمر", col_balance: "الرصيد",
    col_products: "عدد المنتجات", col_active: "عضو نشط",
    risk_tag_high: "خطر مرتفع", risk_tag_low: "منخفض",
    pred_yes: "نعم", pred_no: "لا",
    sum_total: "إجمالي العملاء", sum_high: "خطر توقف مرتفع", sum_rate: "نسبة الخطر",
    sum_low: "خطر منخفض", sum_thresh: "عتبة القرار", sum_file: "الملف المُعالج",
    note_shown: "عرض أول {shown} من أصل {total} عميل — يمكنك تحميل النتائج كاملة.",
    note_all: "إجمالي {total} عميل — يمكنك تحميل النتائج كاملة.",
    btn_download: "تحميل النتائج (CSV)",
    err_ext: "الرجاء رفع ملف بصيغة CSV أو Excel",
    err_unexpected: "حدث خطأ غير متوقع",
    footer: "ChurnScope · نظام التنبؤ بتوقف العملاء — مبني بـ Python / Flask / XGBoost",
    switch_lang: "English",
  },
};

let THRESHOLD = 0.24;
let gaugeChart = null;

const state = {
  lang: localStorage.getItem("churn_lang") === "ar" ? "ar" : "en",
  lastProb: null,
  lastRisk: null,
  lastBatch: null, // {rows, cols, meta}
  fileName: "",
};

const t = (k) => (I18N[state.lang][k] ?? I18N.en[k] ?? k);

function applyLanguage() {
  const lang = state.lang;
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
  document.querySelectorAll("[data-i18n-html]").forEach((el) => (el.innerHTML = t(el.dataset.i18nHtml)));
  if (state.lastProb !== null && state.lastRisk) {
    const badge = $("#riskBadge");
    badge.className = "risk-badge " + (state.lastRisk === "high" ? "high" : "low");
    badge.textContent = state.lastRisk === "high" ? t("risk_high") : t("risk_low");
  }
  if (state.lastProb !== null) renderGauge(state.lastProb);
  if (state.lastBatch) renderBatch(state.lastBatch, state.fileName);
}

$("#langToggle").addEventListener("click", () => {
  state.lang = state.lang === "en" ? "ar" : "en";
  localStorage.setItem("churn_lang", state.lang);
  applyLanguage();
});

/* ---------- init ---------- */
applyLanguage();
fetch("/health")
  .then((r) => r.json())
  .then((d) => {
    THRESHOLD = d.threshold;
    $("#thrBadge").textContent = THRESHOLD.toFixed(2);
    $("#thrInfo").textContent = THRESHOLD.toFixed(2);
  })
  .catch(() => {});

/* ---------- tabs ---------- */
$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    $$(".tab").forEach((tb) => tb.classList.remove("active"));
    $("#tab-" + btn.dataset.tab).classList.add("active");
  });
});

/* ---------- gauge ---------- */
function renderGauge(prob) {
  const ctx = $("#gauge");
  if (gaugeChart) gaugeChart.destroy();
  const isHigh = prob >= THRESHOLD;
  const color = isHigh ? "#f87171" : "#34d399";
  const label = t("gauge_label");
  const textPlugin = {
    id: "centerText",
    afterDraw(chart) {
      const { ctx: c } = chart;
      const x = chart.getDatasetMeta(0).data[0].x;
      const y = chart.getDatasetMeta(0).data[0].y;
      c.save();
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.font = "900 30px Tajawal";
      c.fillStyle = "#eef1ff";
      c.fillText((prob * 100).toFixed(0) + "%", x, y - 8);
      c.font = "600 12px Tajawal";
      c.fillStyle = "#a7aed8";
      c.fillText(label, x, y + 18);
      c.restore();
    },
  };
  gaugeChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [prob, 1 - prob],
        backgroundColor: [color, "rgba(255,255,255,.09)"],
        borderWidth: 0,
      }],
    },
    options: { cutout: "74%", responsive: true, plugins: { legend: { display: false }, tooltip: { enabled: false } } },
    plugins: [textPlugin],
  });
}

/* ---------- single form ---------- */
$("#singleForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = $("#singleBtn");
  btn.disabled = true;
  btn.querySelector("span.btn-ico").textContent = "…";
  try {
    const payload = {
      age: parseFloat($("#age").value),
      geography: $("#geography").value,
      gender: $("#gender").value,
      balance: parseFloat($("#balance").value || 0),
      num_of_products: parseInt($("#numOfProducts").value, 10),
      is_active_member: parseInt($("#isActiveMember").value, 10),
    };
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || t("err_unexpected"));
    state.lastProb = data.probability;
    state.lastRisk = data.risk;
    $("#probPct").textContent = (data.probability * 100).toFixed(1);
    const badge = $("#riskBadge");
    badge.className = "risk-badge " + (data.risk === "high" ? "high" : "low");
    badge.textContent = data.risk === "high" ? t("risk_high") : t("risk_low");
    $("#singleResult").classList.remove("hidden");
    renderGauge(data.probability);
    $("#singleResult").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.querySelector("span.btn-ico").textContent = "⚡";
  }
});

$("#resetBtn").addEventListener("click", () => {
  $("#singleForm").reset();
  $("#age").value = "38";
  $("#singleResult").classList.add("hidden");
  state.lastProb = null;
  state.lastRisk = null;
});

/* ---------- dropzone / upload ---------- */
const dz = $("#dropzone");
const input = $("#fileInput");

dz.addEventListener("click", () => input.click());
input.addEventListener("change", () => {
  if (input.files.length) handleFile(input.files[0]);
  input.value = "";
});
["dragover", "dragenter"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("dragover"); })
);
dz.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

async function handleFile(file) {
  const ok = /\.(csv|xlsx|xls)$/i.test(file.name);
  if (!ok) return showError(t("err_ext"));
  $("#errorBox").classList.add("hidden");
  $("#batchResult").classList.add("hidden");
  $("#progress").classList.remove("hidden");
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/predict_file", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || t("err_unexpected"));
    state.lastBatch = {
      rows: data.rows,
      cols: data.columns,
      meta: {
        total: data.total,
        high: data.high_risk,
        pct: data.high_risk_pct,
        threshold: data.threshold,
      },
    };
    state.fileName = file.name;
    renderBatch(state.lastBatch, state.fileName);
  } catch (err) {
    showError(err.message);
  } finally {
    $("#progress").classList.add("hidden");
  }
}

function showError(msg) {
  const box = $("#errorBox");
  box.textContent = msg;
  box.classList.remove("hidden");
}

function colLabel(col) {
  const map = {
    Churn_Probability: "col_conv", Churn_Prediction: "col_pred", Risk: "col_risk",
    Geography: "col_country", Gender: "col_gender", Age: "col_age",
    Balance: "col_balance", NumOfProducts: "col_products", IsActiveMember: "col_active",
  };
  return map[col] ? t(map[col]) : col;
}

function renderBatch(data, fname) {
  const meta = data.meta;
  const sum = $("#summary");
  sum.innerHTML = `
    <div class="sum-card"><div class="num">${meta.total.toLocaleString("en-US")}</div><div class="lbl">${t("sum_total")}</div></div>
    <div class="sum-card danger"><div class="num">${meta.high.toLocaleString("en-US")}</div><div class="lbl">${t("sum_high")}</div></div>
    <div class="sum-card warn"><div class="num">${meta.pct.toFixed(1)}%</div><div class="lbl">${t("sum_rate")}</div></div>
    <div class="sum-card ok"><div class="num">${(100 - meta.pct).toFixed(1)}%</div><div class="lbl">${t("sum_low")}</div></div>
    <div class="sum-card"><div class="num">${Number(meta.threshold).toFixed(2)}</div><div class="lbl">${t("sum_thresh")}</div></div>
    <div class="sum-card"><div class="num" style="font-size:15px">${fname}</div><div class="lbl">${t("sum_file")}</div></div>`;

  $("#tableHead").innerHTML = data.cols.map((c) => `<th>${colLabel(c)}</th>`).join("");

  const body = $("#tableBody");
  body.innerHTML = "";
  const shown = Math.min(data.rows.length, 100);
  data.rows.slice(0, shown).forEach((r) => {
    const tr = document.createElement("tr");
    if (String(r.Risk) === "high") tr.className = "hr";
    tr.innerHTML = data.cols
      .map((c) => {
        let v = r[c];
        if (v === null || v === undefined) return "<td>—</td>";
        if (c === "Churn_Probability") return `<td>${(v * 100).toFixed(2)}%</td>`;
        if (c === "Risk") {
          const cls = v === "high" ? "high" : "low";
          return `<td><span class="tag ${cls}">${v === "high" ? t("risk_tag_high") : t("risk_tag_low")}</span></td>`;
        }
        if (c === "Churn_Prediction") return `<td>${v === 1 ? t("pred_yes") : t("pred_no")}</td>`;
        return `<td>${v}</td>`;
      })
      .join("");
    body.appendChild(tr);
  });

  const total = data.rows.length.toLocaleString("en-US");
  $("#tableNote").textContent =
    data.rows.length > shown
      ? t("note_shown").replace("{shown}", shown).replace("{total}", total)
      : t("note_all").replace("{total}", total);

  $("#batchResult").classList.remove("hidden");
  $("#batchResult").scrollIntoView({ behavior: "smooth" });
}

/* ---------- download CSV ---------- */
$("#downloadBtn").addEventListener("click", () => {
  if (!state.lastBatch) return;
  const { rows, cols } = state.lastBatch;
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    let s = String(v);
    if (/[",\n]/.test(s)) s = '"' + s.replace(/"/g, '""') + '"';
    return s;
  };
  const lines = [cols.join(",")];
  rows.forEach((r) => lines.push(cols.map((c) => esc(r[c])).join(",")));
  const blob = new Blob(["\ufeff" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "churn_predictions.csv";
  a.click();
  URL.revokeObjectURL(a.href);
});