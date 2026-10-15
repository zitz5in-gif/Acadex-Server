from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import deepl
import re

app = FastAPI(title="Acadex Ultimate Academic SPSS & Translate API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(".App_Data/Uploads")
DOWNLOAD_DIR = Path(".App_Data/Downloads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

LIKERT_MAP = {
    "أوافق بشدة": 5, "موافق بشدة": 5, "strongly agree": 5, "اتفق تماما": 5, "اتفق بشدة": 5, "دائما": 5, "تماما": 5,
    "أوافق": 4, "موافق": 4, "agree": 4, "اتفق": 4, "غالبا": 4, "نعم": 4,
    "محايد": 3, "neutral": 3, "متوسط": 3, "احيانا": 3, "أحيانا": 3, "إلى حد ما": 3, "الى حد ما": 3,
    "لا أوافق": 2, "غير موافق": 2, "disagree": 2, "لا اتفق": 2, "نادرا": 2, "لا": 2,
    "لا أوافق بشدة": 1, "غير موافق بشدة": 1, "strongly disagree": 1, "لا اتفق تماما": 1, "ابدا": 1, "أبدا": 1, "مطلقا": 1
}

def convert_text_to_numeric(val):
    if pd.isna(val) or str(val).strip() == "": return np.nan
    val_str = str(val).strip()
    try: 
        num = float(val_str)
        if 1 <= num <= 5: return num
    except: pass
    
    val_lower = val_str.lower()
    
    # فلترة الذكاء الاصطناعي لحل تداخل (متوسط) مع (متوسطة) أو كلمات الديموغرافية
    ignore_words = ["متوسطة", "بكالوريوس", "دبلوم", "ماجستير", "دكتوراه", "سنة", "سنوات"]
    if any(iw in val_lower for iw in ignore_words):
        return np.nan

    # البحث عن المطابقة
    for key, num in LIKERT_MAP.items():
        if key == val_lower or key in val_lower: return num
    return np.nan

def get_approval_degree(mean_val):
    if mean_val >= 3.67: return "عالية"
    elif mean_val >= 2.34: return "متوسطة"
    else: return "منخفضة"

def extract_base_name(text):
    """تنظيف اسم العمود ليظهر بشكل أنيق في الجداول بدون أرقام أو رموز"""
    clean_text = re.sub(r'^\d+[\-\.\)]?\s*', '', str(text).strip())
    clean_text = clean_text.replace('\n', ' ').strip()
    if len(clean_text) > 60:
        return clean_text[:60] + "..."
    return clean_text

@app.post("/api/translate/upload")
async def upload_and_translate(file: UploadFile = File(...), targetLang: str = Form("ar"), userApiKey: str = Form(...)):
    try:
        translator = deepl.Translator(userApiKey)
        file_ext = Path(file.filename).suffix.lower()
        input_filepath = UPLOAD_DIR / file.filename
        with open(input_filepath, "wb") as buffer:
            buffer.write(await file.read())
        translated_filename = f"Translated_{Path(file.filename).stem}{file_ext}"
        output_filepath = DOWNLOAD_DIR / translated_filename
        lang_mapping = {"ar": "AR", "en": "EN-US"}
        translator.translate_document_from_filepath(str(input_filepath), str(output_filepath), target_lang=lang_mapping.get(targetLang, "AR"))
        return {"success": True, "translated_text": "✅ تمت ترجمة المستند بنجاح.", "download_url": f"http://127.0.0.1:8000/api/download/{translated_filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/statistics")
async def analyze_statistics(file: UploadFile = File(...), tableType: str = Form("descriptive"), dimSplit: str = Form(""), varSplit: str = Form("")):
    try:
        file_ext = Path(file.filename).suffix.lower()
        filepath = UPLOAD_DIR / file.filename
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        df = None
        if file_ext in ['.xlsx', '.xls']: df = pd.read_excel(filepath)
        elif file_ext == '.csv':
            try: df = pd.read_csv(filepath, encoding='utf-8')
            except: df = pd.read_csv(filepath, encoding='windows-1256')

        if df is None or df.empty: raise HTTPException(status_code=400, detail="يرجى رفع ملف بيانات صالح.")

        df = df.dropna(axis=0, how='all').reset_index(drop=True)
        unnamed_count = sum(1 for c in df.columns if 'unnamed' in str(c).lower())
        if unnamed_count > len(df.columns) / 2 and len(df) > 0:
            df.columns = df.iloc[0].astype(str)
            df = df[1:].reset_index(drop=True)

        df.columns = [str(c).strip().replace('\n', ' ').replace('\r', '') for c in df.columns]
        df = df.dropna(axis=1, how='all')
        
        original_n = len(df)
        if original_n == 0: raise HTTPException(status_code=400, detail="الملف فارغ.")

        demo_cols = []
        likert_cols = []
        
        # فرز ذكي للأعمدة لحل مشكلة الديموغرافية
        for c in df.columns:
            c_str = str(c).strip().lower()
            if c_str in ["ت", "م", "رقم", "تسلسل", "id", "no", "seq", "n", "طابع زمني"]: continue
            sample = df[c].dropna().astype(str)
            if len(sample) == 0: continue
            
            converted = sample.apply(convert_text_to_numeric)
            if converted.notna().mean() >= 0.4: likert_cols.append(c)
            else: demo_cols.append(c)

        if tableType in ["gender", "age", "education", "experience"]:
            kw_map = {"gender": ["جنس", "نوع"], "age": ["عمر", "سن"], "education": ["مؤهل", "تعليم", "شهادة", "تحصيل"], "experience": ["خدمة", "سنوات", "خبرة", "حدمة"]}
            col_name = None
            for c in demo_cols:
                if any(kw in str(c).lower() for kw in kw_map[tableType]): col_name = c; break
            
            if col_name is None:
                for c in demo_cols:
                    sample = df[c].dropna().astype(str)
                    if len(sample) == 0: continue
                    if tableType == "gender" and sample.str.contains('ذكر|أنثى|انثى', regex=True).sum() > len(sample) * 0.2: col_name = c; break
                    elif tableType == "education" and sample.str.contains('بكالوريوس|دبلوم|ماجستير|دكتوراه|اعدادية|إعدادية|متوسطة', regex=True).sum() > len(sample) * 0.2: col_name = c; break
                    elif tableType == "age" and sample.str.contains('سنة|سن|عام|أقل|اقل|فأكثر', regex=True).sum() > len(sample) * 0.2 and not sample.str.contains('خدمة|خبرة|حدمة', regex=True).any(): col_name = c; break
                    elif tableType == "experience" and sample.str.contains('خدمة|خبرة|سنة|سنوات|حدمة', regex=True).sum() > len(sample) * 0.2: col_name = c; break

            if col_name is not None:
                col_data = df[col_name].dropna()
                col_data = col_data[col_data != col_name] 
                valid_n = len(col_data)
                rejected_n = original_n - valid_n
                counts = col_data.value_counts()
                total = counts.sum()
                pcts = [round((c / total) * 100, 1) for c in counts]
                diff = round(100.0 - sum(pcts), 1)
                if diff != 0 and len(pcts) > 0: pcts[0] = round(pcts[0] + diff, 1)
                
                table_data = []
                for i, (cat, count) in enumerate(counts.items()):
                    table_data.append({"category": str(cat).replace('\n', ' ').strip(), "count": int(count), "percentage": f"{pcts[i]}%"})
                title_map = {"gender": "توزيع العينة حسب الجنس", "age": "توزيع العينة حسب العمر", "education": "توزيع العينة حسب المؤهل العلمي", "experience": "توزيع العينة حسب سنوات الخدمة"}
                return {"success": True, "type": "demographic", "title": title_map[tableType], "valid_n": valid_n, "rejected_n": rejected_n, "data": table_data}
            else: raise HTTPException(status_code=400, detail=f"لم يتم العثور على أي بيانات تخص '{tableType}'.")

        if len(likert_cols) == 0: raise HTTPException(status_code=400, detail="لم يعثر النظام على أسئلة استبيان (1-5).")
        analysis_df = df[likert_cols].copy()
        for col in likert_cols: analysis_df[col] = analysis_df[col].apply(convert_text_to_numeric)
        analysis_df = analysis_df.dropna(how='any')
        valid_n = len(analysis_df)
        rejected_n = original_n - valid_n
        if valid_n < 2: raise HTTPException(status_code=400, detail=f"الاستبانات المكتملة {valid_n} فقط، لا تكفي للتحليل.")

        # استخراج أسماء الأبعاد والمتغيرات الحقيقية من الملف
        dim_splits = []
        if dimSplit and dimSplit.strip():
            parts = [p.strip() for p in dimSplit.split(',') if p.strip().isdigit()]
            if len(parts) == 1:
                chunk = int(parts[0])
                if chunk > 0:
                    dim_splits = [chunk] * (len(likert_cols) // chunk)
                    if len(likert_cols) % chunk != 0: dim_splits.append(len(likert_cols) % chunk)
            else: dim_splits = [int(p) for p in parts if int(p) > 0]
        else: dim_splits = [len(likert_cols)]

        dimensions = []
        col_idx = 0
        for i, d_len in enumerate(dim_splits):
            cols = likert_cols[col_idx : col_idx+d_len]
            if cols:
                # سحب اسم البعد من أول سؤال فيه إذا أمكن
                base_name = extract_base_name(cols[0])
                # إذا كان الاسم طويلاً جداً أو مجرد رقم، نسميه بعد رقم كذا
                if len(base_name) > 40 or base_name.isdigit():
                    name = f"البعد {i+1}"
                else:
                    # أخذ أول 4 كلمات من السؤال كاسم للبعد
                    name = " ".join(base_name.split()[:4])
                dimensions.append({"name": name, "cols": cols})
            col_idx += d_len

        var_splits = []
        if varSplit and varSplit.strip():
            parts = [p.strip() for p in varSplit.split(',') if p.strip().isdigit()]
            if len(parts) == 1:
                chunk = int(parts[0])
                if chunk > 0:
                    var_splits = [chunk] * (len(dimensions) // chunk)
                    if len(dimensions) % chunk != 0: var_splits.append(len(dimensions) % chunk)
            else: var_splits = [int(p) for p in parts if int(p) > 0]
        else: var_splits = [len(dimensions)]

        variables = []
        dim_idx = 0
        for i, v_len in enumerate(var_splits):
            dims = dimensions[dim_idx : dim_idx+v_len]
            if dims:
                # تسمية المتغيرات كمتغير مستقل وتابع للتوضيح إذا كانا 2 فقط
                name = f"المتغير {i+1}"
                if len(var_splits) == 2:
                    name = "المتغير المستقل" if i == 0 else "المتغير التابع"
                variables.append({"name": name, "dims": dims})
            dim_idx += v_len

        computed_df = pd.DataFrame()
        dim_names = []
        for d in dimensions:
            d_name = d["name"]
            computed_df[d_name] = analysis_df[d["cols"]].mean(axis=1)
            dim_names.append(d_name)
            
        var_names = []
        for v in variables:
            v_name = v["name"]
            v_cols = []
            for d in v["dims"]: v_cols.extend(d["cols"])
            computed_df[v_name] = analysis_df[v_cols].mean(axis=1)
            var_names.append(v_name)

        if tableType == "descriptive":
            table_data = []
            for var in variables:
                for dim in var["dims"]:
                    table_data.append({"dimension": dim['name'], "mean": "", "std": "", "approval": "", "is_header": True})
                    for col in dim["cols"]:
                        m = float(analysis_df[col].mean())
                        s = float(analysis_df[col].std(ddof=1)) if valid_n > 1 else 0.0
                        clean_col = extract_base_name(col)
                        table_data.append({"dimension": f" - {clean_col}", "mean": round(m, 2), "std": round(s, 2), "approval": get_approval_degree(m), "is_header": False})
                    
                    dim_series = analysis_df[dim["cols"]].mean(axis=1)
                    dm = float(dim_series.mean())
                    ds = float(dim_series.std(ddof=1)) if valid_n > 1 else 0.0
                    table_data.append({"dimension": f"المتوسط العام ({dim['name']})", "mean": round(dm, 2), "std": round(ds, 2), "approval": get_approval_degree(dm), "is_summary": True})
                
                if varSplit.strip():
                    v_cols = []
                    for d in var["dims"]: v_cols.extend(d["cols"])
                    var_series = analysis_df[v_cols].mean(axis=1)
                    vm = float(var_series.mean())
                    vs = float(var_series.std(ddof=1)) if valid_n > 1 else 0.0
                    table_data.append({"dimension": f"الإجمالي لـ ({var['name']})", "mean": round(vm, 2), "std": round(vs, 2), "approval": get_approval_degree(vm), "is_total": True})

            return {"success": True, "type": "descriptive_academic", "valid_n": valid_n, "rejected_n": rejected_n, "data": table_data}

        elif tableType == "correlation":
            if len(var_names) < 2: raise HTTPException(status_code=400, detail="تحليل الارتباط يتطلب متغيرين على الأقل (مستقل وتابع).")
            dep_var = variables[-1]
            indep_var = variables[0]
            dep_cols_names = [d["name"] for d in dep_var["dims"]] + [dep_var["name"]]
            indep_cols_names = [d["name"] for d in indep_var["dims"]] + [indep_var["name"]]
            matrix_data = []
            for indep_name in indep_cols_names:
                row_data = {"indep_name": indep_name, "correlations": {}}
                for dep_name in dep_cols_names:
                    try: r, _ = stats.pearsonr(computed_df[indep_name], computed_df[dep_name])
                    except: r = 0.0
                    row_data["correlations"][dep_name] = round(r, 3)
                matrix_data.append(row_data)
            return {"success": True, "type": "correlation_matrix", "valid_n": valid_n, "rejected_n": rejected_n, "dep_headers": dep_cols_names, "data": matrix_data}

        elif tableType == "simple_regression":
            if len(var_names) < 2: raise HTTPException(status_code=400, detail="الانحدار يتطلب متغيرين على الأقل.")
            dep_name = var_names[-1]
            indep_var = variables[0]
            table_data = []
            for dim in indep_var["dims"]:
                indep_name = dim["name"]
                try: slope, intercept, r_value, p_value, std_err = stats.linregress(computed_df[indep_name], computed_df[dep_name])
                except: slope, intercept, r_value, p_value, std_err = 0.0, 0.0, 0.0, 1.0, 0.0
                r2 = r_value ** 2
                f_val = (slope ** 2) / (std_err ** 2) if std_err > 0 else 0.0
                table_data.append({"indep": indep_name, "B": round(slope, 3), "R2": round(r2, 3), "F": round(f_val, 3)})
            try: slope, intercept, r_value, p_value, std_err = stats.linregress(computed_df[indep_var["name"]], computed_df[dep_name])
            except: slope, intercept, r_value, p_value, std_err = 0.0, 0.0, 0.0, 1.0, 0.0
            r2 = r_value ** 2
            f_val = (slope ** 2) / (std_err ** 2) if std_err > 0 else 0.0
            table_data.append({"indep": f"إجمالي {indep_var['name']}", "B": round(slope, 3), "R2": round(r2, 3), "F": round(f_val, 3), "is_summary": True})
            return {"success": True, "type": "effect_table", "dep_name": dep_name, "valid_n": valid_n, "rejected_n": rejected_n, "data": table_data}

        elif tableType == "multiple_regression":
            if len(var_names) < 2: raise HTTPException(status_code=400, detail="الانحدار المتعدد يتطلب متغير تابع.")
            dep_name = var_names[-1]
            indep_dims = [d["name"] for d in variables[0]["dims"]]
            if len(indep_dims) < 2: raise HTTPException(status_code=400, detail="الانحدار المتعدد يتطلب أكثر من بعد مستقل.")
            Y = computed_df[dep_name].values.astype(float)
            X = computed_df[indep_dims].values.astype(float)
            X_with_const = np.column_stack((np.ones(len(X)), X))
            try:
                beta = np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T @ Y
                Y_pred = X_with_const @ beta
                ss_tot = np.sum((Y - np.mean(Y))**2)
                ss_res = np.sum((Y - Y_pred)**2)
                ss_reg = ss_tot - ss_res
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                r_val = np.sqrt(r2)
                df_reg = len(indep_dims)
                df_res = len(Y) - df_reg - 1
                df_tot = len(Y) - 1
                ms_reg = ss_reg / df_reg if df_reg > 0 else 0.0
                ms_res = ss_res / df_res if df_res > 0 else 0.0
                f_stat = ms_reg / ms_res if ms_res > 0 else 0.0
                p_val_model = 1.0 - stats.f.cdf(f_stat, df_reg, df_res) if ms_res > 0 else 1.0
                var_b = ms_res * np.linalg.pinv(X_with_const.T @ X_with_const)
                se_b = np.sqrt(np.diag(var_b))
                t_stats = beta / se_b
                p_vals_t = [2 * (1 - stats.t.cdf(np.abs(t), df_res)) for t in t_stats]

                anova_data = [
                    {"model": "Regression", "ss": round(ss_reg, 3), "df": df_reg, "ms": round(ms_reg, 3), "f": round(f_stat, 3), "sig": "<0.001" if p_val_model < 0.001 else round(p_val_model, 3)},
                    {"model": "Residual", "ss": round(ss_res, 3), "df": df_res, "ms": round(ms_res, 3), "f": "", "sig": ""},
                    {"model": "Total", "ss": round(ss_tot, 3), "df": df_tot, "ms": "", "f": "", "sig": ""}
                ]
                coeff_data = []
                for i, dim_name in enumerate(indep_dims):
                    coeff_data.append({
                        "variable": dim_name,
                        "R": round(r_val, 3) if i == 0 else "", "R2": round(r2, 3) if i == 0 else "",
                        "B": round(beta[i+1], 3), "std_error": round(se_b[i+1], 3),
                        "Beta_std": round(beta[i+1] * (np.std(X[:, i]) / np.std(Y)), 3),
                        "T": round(t_stats[i+1], 3), "sig": "<0.001" if p_vals_t[i+1] < 0.001 else round(p_vals_t[i+1], 3)
                    })
                return {"success": True, "type": "multiple_regression_academic", "valid_n": valid_n, "rejected_n": rejected_n, "anova": anova_data, "coeff": coeff_data}
            except Exception as ex: raise HTTPException(status_code=500, detail=f"حدث خطأ رياضي: {str(ex)}")

        else: raise HTTPException(status_code=400, detail="نوع الجدول غير معروف.")

    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=f"خطأ غير متوقع: {str(e)}")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = DOWNLOAD_DIR / filename
    if file_path.exists(): return FileResponse(path=file_path, filename=filename)
    raise HTTPException(status_code=404, detail="الملف غير موجود.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)