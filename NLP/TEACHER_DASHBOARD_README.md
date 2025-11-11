# 🎓 Teacher Dashboard - Batch Grading System

## Overview

The Teacher Dashboard is an extension of the ASAG system that allows teachers to **grade entire classes at once** by uploading a CSV/Excel file with all student responses. The system automatically generates:

- ✅ Individual detailed scoresheets for each student (Excel format)
- ✅ Class summary report with statistics
- ✅ Per-question breakdown with similarity scores
- ✅ Automated grade assignment (A/B/C/D)

---

## 🚀 Quick Start

### 1. Access the Dashboard

Open your browser and go to:
```
http://localhost:5000/teacher
```

### 2. Prepare Your File

Export your Google Forms responses or create a CSV/Excel file with this format:

| Name | Roll Number | Q1 | Q2 | Q3 |
|------|-------------|----|----|-----|
| John Doe | 101 | Student answer 1... | Student answer 2... | Student answer 3... |
| Jane Smith | 102 | Student answer 1... | Student answer 2... | Student answer 3... |

**Required Columns:**
- `Name` - Student's full name
- `Roll Number` - Student ID/Roll number
- `Q1`, `Q2`, `Q3`, etc. - Question columns (any names work, e.g., "Question 1", "Biology Q1")

### 3. Upload and Grade

1. **Upload File:** Drag & drop or click to browse your CSV/Excel file
2. **Enter Model Answers:** Fill in the ideal answer for each detected question
3. **Click "Grade All Students":** Wait for processing (10-30 seconds depending on class size)
4. **Download Results:** Get individual scoresheets + class summary

---

## 📋 File Format Guide

### Google Forms Export

1. Open your Google Form responses
2. Click the **three dots** (⋮) menu
3. Select **"Download responses (.csv)"**
4. Upload the downloaded file to the dashboard

### Manual CSV Format

```csv
Name,Roll Number,Q1,Q2,Q3
John Doe,101,Photosynthesis is...,The water cycle...,DNA contains...
Jane Smith,102,Plants use sunlight...,Water evaporates...,Genes are...
Bob Johnson,103,Cells need light...,Rain falls...,Chromosomes have...
```

### Excel Format

- Use `.xlsx` or `.xls` files
- First row must be headers
- Each subsequent row is one student

---

## 📊 Generated Reports

### Individual Student Scoresheets

Each student gets a detailed Excel file containing:

**Header Section:**
- Student Name
- Roll Number
- Date & Time
- Total Score (e.g., 7/9)
- Percentage (e.g., 77.8%)
- Grade (A/B/C/D)

**Question Breakdown Table:**
| Question | Score | Max | Similarity | Comments |
|----------|-------|-----|------------|----------|
| Q1 | 3 | 3 | 92% | Excellent! All key concepts covered. |
| Q2 | 2 | 3 | 78% | Good. Most concepts covered. |
| Q3 | 2 | 3 | 71% | Good. Most concepts covered. |

**Detailed Answers Section:**
- Shows model answer vs student answer for each question
- Color-coded scores (Green=3, Yellow=2, Red=0-1)

**Filename Format:**
```
John_Doe_101_scoresheet.xlsx
```

### Class Summary Report

A single Excel file with:

**Student Overview Table:**
| Name | Roll Number | Total Score | Max Score | Percentage | Grade |
|------|-------------|-------------|-----------|------------|-------|
| John Doe | 101 | 8 | 9 | 88.9% | A |
| Jane Smith | 102 | 7 | 9 | 77.8% | B |
| Bob Johnson | 103 | 5 | 9 | 55.6% | C |

**Class Statistics:**
- Average Score
- Highest Score
- Lowest Score
- Grade Distribution

**Filename Format:**
```
class_summary_20251103_201530.xlsx
```

---

## 🎯 Scoring System

### Score Scale: 0-3 (per question)

| Score | Description | Similarity Range | Color |
|-------|-------------|------------------|-------|
| **3** | Excellent | 85%+ | 🟢 Green |
| **2** | Good | 70-85% | 🟡 Yellow |
| **1** | Fair | 50-70% | 🟠 Orange |
| **0** | Poor | < 50% | 🔴 Red |

### Grade Assignment (overall)

| Grade | Percentage Range |
|-------|------------------|
| **A** | 85% and above |
| **B** | 70-84% |
| **C** | 50-69% |
| **D** | Below 50% |

### Grading Algorithm

For each question, the system uses:
1. **TF-IDF Model** (keyword matching)
2. **SBERT Cosine Similarity** (semantic understanding)
3. **Ensemble Score** = 50% TF-IDF + 50% Cosine Score

This combined approach ensures both keyword accuracy and semantic understanding are evaluated.

---

## 💡 Best Practices

### Writing Model Answers

✅ **Good Model Answer:**
```
Photosynthesis is the process by which plants convert sunlight into chemical energy. 
Plants use chlorophyll in their leaves to absorb light energy, which is then used to 
convert carbon dioxide and water into glucose and oxygen.
```

❌ **Poor Model Answer:**
```
Plants make food.
```

**Tips:**
- Include all key concepts
- Use clear, complete sentences
- Be specific and detailed
- Length: 2-4 sentences typically works best

### Preparing Student Data

✅ **Do:**
- Use consistent column names
- Include Name and Roll Number columns
- Keep question column names short (Q1, Q2, etc.)
- Remove any timestamp columns from Google Forms

❌ **Don't:**
- Mix different question formats in one file
- Include special characters in names
- Have empty required fields

---

## 🔧 Technical Details

### File Size Limits

- **Maximum file size:** 16MB
- **Maximum students:** ~1000 (depends on answer length)
- **Processing time:** 10-30 seconds for 30 students

### Supported File Formats

- `.csv` - Comma-separated values
- `.xlsx` - Excel 2007+
- `.xls` - Excel 97-2003

### Storage

Generated files are saved in:
```
E:\NLP\results\
```

Files persist until manually deleted or server restart.

---

## 📈 API Endpoints

### POST `/batch-grade`

**Purpose:** Process batch grading

**Request:**
- `file`: CSV/Excel file (multipart/form-data)
- `model_answers`: JSON string mapping question columns to answers

**Response:**
```json
{
  "ok": true,
  "message": "Successfully graded 30 students",
  "summary_file": "class_summary_20251103.xlsx",
  "individual_files": ["John_Doe_101_scoresheet.xlsx", ...],
  "total_students": 30,
  "results": [...]
}
```

### GET `/download/<filename>`

**Purpose:** Download generated result files

**Example:**
```
GET /download/class_summary_20251103_201530.xlsx
```

### GET `/results`

**Purpose:** List all generated result files

**Response:**
```json
{
  "ok": true,
  "files": [
    {
      "filename": "class_summary_20251103.xlsx",
      "size": 15234,
      "created": 1699034130
    }
  ]
}
```

---

## 🐛 Troubleshooting

### Issue: "No question columns detected"

**Problem:** System couldn't find question columns

**Solution:**
- Make sure columns have names like Q1, Q2, Question1, etc.
- Remove email/timestamp columns
- Check that first row contains headers

### Issue: File upload fails

**Problem:** File too large or wrong format

**Solution:**
- Check file size (must be < 16MB)
- Verify file is CSV or Excel format
- Try exporting Google Forms again

### Issue: Low scores for good answers

**Problem:** Model answers might be too short

**Solution:**
- Make model answers more detailed
- Include all key concepts
- Use 2-4 complete sentences

### Issue: "Cannot read file"

**Problem:** File encoding issues

**Solution:**
- Save CSV as UTF-8 encoding
- Try Excel format instead
- Remove special characters from text

---

## 🔒 Privacy & Security

### Data Handling

- ✅ Files are processed locally on your server
- ✅ No data sent to external APIs
- ✅ Uploaded files deleted after processing
- ✅ Results stored locally in `results/` folder

### Recommendations

- Don't expose server publicly without authentication
- Regularly clean up the `results/` folder
- Use HTTPS in production
- Add login system for multi-teacher use

---

## 🆚 Single vs Batch Grading

| Feature | Single Grading | Batch Grading |
|---------|---------------|---------------|
| **Use Case** | Quick checks, demos | Entire class grading |
| **Input** | Manual text entry | CSV/Excel upload |
| **Output** | JSON response | Excel scoresheets |
| **Time** | 1-2 seconds | 10-30 seconds |
| **Best For** | Testing, examples | Homework, quizzes |

---

## 📚 Examples

### Example 1: Science Quiz

**File:** `science_quiz_responses.csv`

```csv
Name,Roll Number,Q1,Q2,Q3
Alice Brown,201,Plants use sunlight and chlorophyll to make glucose through photosynthesis.,The water cycle involves evaporation condensation and precipitation.,Mitosis is cell division that creates two identical daughter cells.
```

**Model Answers:**
- Q1: "Photosynthesis is the process by which plants convert sunlight into energy using chlorophyll."
- Q2: "The water cycle describes how water evaporates, condenses into clouds, and falls as precipitation."
- Q3: "Mitosis is the process of cell division resulting in two genetically identical daughter cells."

**Expected Scores:**
- Alice: Q1=3, Q2=3, Q3=3 → Total 9/9 (100%) → Grade A

### Example 2: History Assignment

**File:** Exported from Google Forms

**Questions:**
- "Explain the causes of the French Revolution"
- "Describe the impact of the Industrial Revolution"
- "What were the main effects of World War I?"

**Processing:**
1. Upload Google Forms CSV
2. Enter detailed model answers (3-4 sentences each)
3. System grades 25 students in ~20 seconds
4. Download 25 individual scoresheets + 1 class summary

---

## 🔮 Future Enhancements

### Planned Features

- [ ] Batch model answer upload (JSON/CSV)
- [ ] Question bank management
- [ ] Historical grading analytics
- [ ] Export grades to LMS (Canvas, Moodle)
- [ ] Email individual scoresheets to students
- [ ] Mobile-responsive dashboard
- [ ] Multi-language support
- [ ] Plagiarism detection
- [ ] Custom rubric creation

---

## 📞 Support

Need help? Check:
1. This documentation
2. Main PROJECT_DOCUMENTATION.md
3. Sample files in the dashboard
4. API examples above

---

**Version:** 1.0  
**Last Updated:** November 3, 2025  
**Status:** Production Ready ✅
