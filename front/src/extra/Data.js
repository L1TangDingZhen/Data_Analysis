import React, { useState } from 'react';

const styles = {
  container: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif',
  },
  card: {
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    padding: '24px',
  },
  header: {
    marginBottom: '24px',
  },
  title: {
    fontSize: '24px',
    fontWeight: '600',
    color: '#111827',
    marginBottom: '8px',
  },
  subtitle: {
    color: '#6B7280',
    fontSize: '14px',
  },
  uploadArea: {
    border: '2px dashed #E5E7EB',
    borderRadius: '8px',
    padding: '32px',
    textAlign: 'center',
    backgroundColor: '#F9FAFB',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  uploadAreaActive: {
    borderColor: '#3B82F6',
    backgroundColor: '#EFF6FF',
  },
  uploadIcon: {
    width: '40px',
    height: '40px',
    color: '#6B7280',
    margin: '0 auto 16px',
  },
  fileInput: {
    display: 'none',
  },
  uploadText: {
    color: '#374151',
    fontSize: '14px',
    marginBottom: '4px',
  },
  uploadSubtext: {
    color: '#6B7280',
    fontSize: '12px',
  },
  loadingContainer: {
    textAlign: 'center',
    padding: '20px',
  },
  spinner: {
    border: '3px solid #f3f3f3',
    borderTop: '3px solid #3498db',
    borderRadius: '50%',
    width: '24px',
    height: '24px',
    animation: 'spin 1s linear infinite',
    margin: '0 auto',
  },
  resultsContainer: {
    marginTop: '24px',
  },
  resultsHeader: {
    padding: '16px',
    backgroundColor: '#F9FAFB',
    borderRadius: '8px 8px 0 0',
    borderBottom: '1px solid #E5E7EB',
  },
  resultsTitle: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#111827',
  },
  resultsSummary: {
    marginTop: '8px',
    fontSize: '14px',
    color: '#6B7280',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    marginTop: '16px',
    fontSize: '14px',
  },
  th: {
    backgroundColor: '#F9FAFB',
    padding: '12px 16px',
    textAlign: 'left',
    color: '#374151',
    fontWeight: '600',
    borderBottom: '1px solid #E5E7EB',
  },
  td: {
    padding: '12px 16px',
    borderBottom: '1px solid #E5E7EB',
    color: '#111827',
  },
  select: {
    width: '100%',
    padding: '8px 12px',
    borderRadius: '6px',
    border: '1px solid #E5E7EB',
    backgroundColor: '#FFFFFF',
    color: '#374151',
    fontSize: '14px',
    outline: 'none',
    transition: 'border-color 0.2s ease',
  },
  selectHover: {
    borderColor: '#3B82F6',
  },
  errorContainer: {
    marginTop: '16px',
    padding: '12px 16px',
    backgroundColor: '#FEF2F2',
    borderRadius: '6px',
    color: '#B91C1C',
    fontSize: '14px',
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
  },
  '@keyframes spin': {
    '0%': { transform: 'rotate(0deg)' },
    '100%': { transform: 'rotate(360deg)' },
  },
  previewContainer: {
    marginTop: '32px',
    padding: '16px',
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
  },
  previewTitle: {
    fontSize: '18px',
    fontWeight: '600',
    color: '#111827',
    marginBottom: '16px',
  },
  previewTable: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '14px',
  },
  previewTh: {
    backgroundColor: '#F9FAFB',
    padding: '12px 16px',
    textAlign: 'left',
    color: '#374151',
    fontWeight: '600',
    borderBottom: '1px solid #E5E7EB',
    whiteSpace: 'nowrap',
  },
  previewTd: {
    padding: '12px 16px',
    borderBottom: '1px solid #E5E7EB',
    color: '#111827',
    maxWidth: '200px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  noDataText: {
    color: '#666',
    fontStyle: 'italic',
    fontSize: '0.9em',
  },
  sampleValue: {
    display: 'inline-block',
    maxWidth: '200px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },

  statisticsContainer: {
    marginTop: '24px',
    padding: '16px',
    backgroundColor: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
  },
  statisticsSection: {
    marginBottom: '16px',
  },
  statisticsTitle: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#111827',
    marginBottom: '8px',
  },
  statisticsItem: {
    marginBottom: '8px',
    padding: '8px',
    backgroundColor: '#f9fafb',
    borderRadius: '4px',
  },

};


const getDisplayType = (type) => {
  const typeMap = {
    'object': 'Text',
    'float64': 'Number (Float)',
    'int64': 'Number (Integer)',
    'datetime64[ns]': 'Date/Time',
    'bool': 'Boolean',
    'category': 'Category',
    'text': 'Text',
    'number': 'Number',
    'datetime': 'Date/Time',
    'boolean': 'Boolean'
  };
  return typeMap[type] || type;
};

const formatSampleValue = (value) => {
  // 处理明确的空值情况
  if (value === null || 
      value === undefined || 
      value === 'nan' || 
      value === 'NaN' || 
      value === '' || 
      value === 'No data available') {
    return 'No data available';
  }

  // 处理数字格式
  if (!isNaN(value) && value !== '') {
    const num = parseFloat(value);
    // 检查是否为整数
    if (Number.isInteger(num)) {
      return num.toString();
    }
    // 如果是小数，保留两位小数
    return Number(num.toFixed(2)).toString();
  }

  // 处理布尔值
  if (value === 'true' || value === 'false') {
    return value;
  }

  // 处理日期格式
  const dateValue = new Date(value);
  if (!isNaN(dateValue) && value.includes('-')) {
    return dateValue.toLocaleDateString();
  }

  // 对于其他所有情况，返回字符串值，并确保去除首尾空格
  const strValue = String(value).trim();
  return strValue || 'No data available';
};


const FileAnalyzer = () => {
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isHovered, setIsHovered] = useState(false);
  const [previewData, setPreviewData] = useState(null);
  const [modifiedColumns, setModifiedColumns] = useState({});


  const handleExport = async () => {
    if (!analysis || !analysis.file_id) return;
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/export/${analysis.file_id}/`, {
        method: 'GET',
      });
      
      if (!response.ok) throw new Error('Export failed');
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'processed_data.csv';
      a.click();
    } catch (err) {
      setError('Failed to export data: ' + err.message);
    }
  };

  const [statistics, setStatistics] = useState(null);
  
  const fetchStatistics = async () => {
    if (!analysis || !analysis.file_id) return;
    
    try {
      const response = await fetch(`http://127.0.0.1:8000/statistics/${analysis.file_id}/`, {
        method: 'GET',
      });
      
      if (!response.ok) throw new Error('Failed to fetch statistics');
      
      const data = await response.json();
      setStatistics(data);
    } catch (err) {
      setError('Failed to fetch statistics: ' + err.message);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.match(/\.(csv|xlsx)$/)) {
      setError('Please upload a CSV or Excel file');
      return;
    }

    setFile(file);
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://127.0.0.1:8000/analyze/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }
      
      const data = await response.json();
      setAnalysis(data);
      setPreviewData(data.preview_data); // 假设后端返回了 preview_data
    } catch (err) {
      console.error('Upload error:', err);
      setError(err.message || 'Failed to analyze file');
    } finally {
      setLoading(false);
    }
  };

  const handleTypeChange = async (column, newType) => {
    setModifiedColumns({
      ...modifiedColumns,
      [column]: newType
    });

    try {
      const response = await fetch('http://127.0.0.1:8000/update-type/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          column,
          new_type: newType,
          file_id: analysis.file_id
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to update type');
      }

      const updatedData = await response.json();
      setPreviewData(updatedData.preview_data);
      
      // 更新分析数据中的类型和样本值
      setAnalysis(prev => ({
        ...prev,
        types: {
          ...prev.types,
          [column]: updatedData.new_type
        },
        samples: {
          ...prev.samples,
          [column]: updatedData.sample_value
        }
      }));

    } catch (err) {
      console.error('Type update error:', err);
      setError('Failed to update column type: ' + err.message);
      // 恢复原来的类型
      setModifiedColumns({
        ...modifiedColumns,
        [column]: analysis.types[column]
      });
    }

  



};

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        <div style={styles.header}>
          <h2 style={styles.title}>Data Type Analyzer</h2>
          <p style={styles.subtitle}>Upload a CSV or Excel file to analyze its data types</p>
        </div>

        <div
          style={{
            ...styles.uploadArea,
            ...(isHovered ? styles.uploadAreaActive : {}),
          }}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
        >
          <input
            type="file"
            onChange={handleFileUpload}
            accept=".csv,.xlsx"
            style={styles.fileInput}
            id="file-upload"
          />
          <label htmlFor="file-upload" style={{ cursor: 'pointer' }}>
            <svg
              style={styles.uploadIcon}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <div style={styles.uploadText}>
              {file ? file.name : 'Click to upload or drag and drop'}
            </div>
            <div style={styles.uploadSubtext}>CSV or Excel files only</div>
          </label>
        </div>

        {loading && (
          <div style={styles.loadingContainer}>
            <div style={styles.spinner} />
            <p style={{ marginTop: '12px', color: '#6B7280' }}>Analyzing file...</p>
          </div>
        )}

        {error && (
          <div style={styles.errorContainer}>
            <svg
              width="20"
              height="20"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            {error}
          </div>
        )}

        {analysis && (
                  <div style={styles.resultsContainer}>
                    <div style={styles.resultsHeader}>
                      <h3 style={styles.resultsTitle}>Analysis Results</h3>
                      <p style={styles.resultsSummary}>
                        Found {Object.keys(analysis.types).length} columns and {analysis.rows} rows
                      </p>
                    </div>
                    
                    <div style={{ overflowX: 'auto' }}>
                      <table style={styles.table}>
                        <thead>
                          <tr>
                            <th style={styles.th}>Column Name</th>
                            <th style={styles.th}>Inferred Type</th>
                            <th style={styles.th}>Sample Value</th>
                            <th style={styles.th}>Change Type</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(analysis.types).map(([column, type]) => (
                            <tr key={column}>
                              <td style={styles.td}>{column}</td>
                              <td style={styles.td}>{getDisplayType(modifiedColumns[column] || type)}</td>
                              <td style={styles.td}>
                                <span style={{
                                  ...styles.sampleValue,
                                  ...(formatSampleValue(analysis.samples[column]) === 'No data available' ? styles.noDataText : {})
                                }}>
                                  {formatSampleValue(analysis.samples[column])}
                                </span>
                              </td>
                              <td style={styles.td}>
                                <select
                                  style={styles.select}
                                  value={modifiedColumns[column] || type}
                                  onChange={(e) => handleTypeChange(column, e.target.value)}
                                >
                                  <option value="text">Text</option>
                                  <option value="number">Number</option>
                                  <option value="datetime">Date/Time</option>
                                  <option value="boolean">Boolean</option>
                                  <option value="category">Category</option>
                                </select>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}


        {analysis && (
          <div style={{
            marginTop: '40px',  // 增加上边距，原来是 mt-4
            paddingBottom: '20px'  // 可选：添加底部内边距
          }}>
            <button 
              onClick={handleExport}
              style={{
                background: 'linear-gradient(to right, #4F46E5, #6366F1)',
                color: 'white',
                padding: '10px 20px',
                borderRadius: '8px',
                border: 'none',
                boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                fontSize: '14px',
                fontWeight: '500',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.transform = 'translateY(-1px)';
                e.currentTarget.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
              }}
            >
              <svg 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2"
                strokeLinecap="round" 
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="7 10 12 15 17 10"/>
                <line x1="12" y1="15" x2="12" y2="3"/>
              </svg>
              Export Processed Data
            </button>
          </div>
        )}


        {statistics && (
          <div className="mt-4 p-4 bg-white rounded shadow">
            <h3 className="text-lg font-semibold">Data Statistics</h3>
            {/* 显示统计信息的具体内容 */}
            {statistics.numeric_columns && (
              <div>
                <h4>Numeric Columns</h4>
                {Object.entries(statistics.numeric_columns).map(([column, stats]) => (
                  <div key={column}>
                    <h5>{column}</h5>
                    <p>Mean: {stats.mean.toFixed(2)}</p>
                    <p>Median: {stats.median.toFixed(2)}</p>
                    <p>Standard Deviation: {stats.std.toFixed(2)}</p>
                    <p>Min: {stats.min}</p>
                    <p>Max: {stats.max}</p>
                  </div>
                ))}
              </div>
            )}

            {statistics.categorical_columns && (
              <div>
                <h4>Categorical Columns</h4>
                {Object.entries(statistics.categorical_columns).map(([column, stats]) => (
                  <div key={column}>
                    <h5>{column}</h5>
                    <p>Unique Values: {stats.unique_count}</p>
                    <div>
                      Value Counts:
                      {Object.entries(stats.value_counts).map(([value, count]) => (
                        <div key={value}>{value}: {count}</div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {statistics.datetime_columns && (
              <div>
                <h4>Date Columns</h4>
                {Object.entries(statistics.datetime_columns).map(([column, stats]) => (
                  <div key={column}>
                    <h5>{column}</h5>
                    <p>First Date: {stats.min_date}</p>
                    <p>Last Date: {stats.max_date}</p>
                    <p>Date Range (days): {stats.date_range}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* 添加数据预览部分 */}
        {previewData && (
          <div style={styles.previewContainer}>
            <h3 style={styles.previewTitle}>
              Data Preview 
              <span style={{
                fontSize: '12px',
                color: '#666',
                marginLeft: '10px',
                fontWeight: 'normal',
              }}>
                (Showing first 5 rows)
              </span>
            </h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={styles.previewTable}>
                <thead>
                  <tr>
                    {Object.keys(previewData[0] || {}).map((header) => (
                      <th key={header} style={styles.previewTh}>
                        <div>{header}</div>
                        {analysis && analysis.types && (
                          <div style={{
                            fontSize: '11px',
                            color: '#666',
                            fontWeight: 'normal',
                            marginTop: '4px'
                          }}>
                            {getDisplayType(analysis.types[header])}
                          </div>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.map((row, index) => (
                    <tr key={index}>
                      {Object.values(row).map((value, cellIndex) => (
                        <td key={cellIndex} style={styles.previewTd}>
                          {formatSampleValue(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileAnalyzer;