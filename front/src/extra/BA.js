import React, { useState } from 'react';

export default function DataTypeInterface() {
  const [data, setData] = useState([
    { column: 'Name', inferredType: 'Text', originalType: 'object' },
    { column: 'Birthdate', inferredType: 'Date', originalType: 'datetime64' },
    { column: 'Score', inferredType: 'Number', originalType: 'float64' },
    { column: 'Grade', inferredType: 'Category', originalType: 'category' }
  ]);

  return (
    <div className="p-4">
      <div className="max-w-4xl mx-auto bg-white shadow rounded-lg">
        <div className="p-6 border-b">
          <h2 className="text-xl font-bold">数据类型自动识别系统</h2>
        </div>
        
        {/* 文件上传区域 */}
        <div className="p-6">
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <div className="mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <button className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
              选择 CSV/Excel 文件
            </button>
            <p className="text-sm text-gray-500 mt-2">支持 .csv 和 .xlsx 格式</p>
          </div>
        </div>

        {/* 数据类型预览表格 */}
        <div className="p-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">列名</th>
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">识别类型</th>
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">原始类型</th>
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {data.map((row, i) => (
                  <tr key={i}>
                    <td className="px-6 py-4 text-sm">{row.column}</td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 text-sm bg-blue-100 text-blue-800 rounded">
                        {row.inferredType}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-500">{row.originalType}</td>
                    <td className="px-6 py-4">
                      <button className="text-sm bg-white border border-gray-300 px-3 py-1 rounded hover:bg-gray-50">
                        修改
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 数据预览区域 */}
      <div className="max-w-4xl mx-auto bg-white shadow rounded-lg mt-8">
        <div className="p-6 border-b">
          <h2 className="text-xl font-bold">数据预览</h2>
        </div>
        <div className="p-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Name</th>
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Birthdate</th>
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Score</th>
                  <th className="px-6 py-3 text-left text-sm font-medium text-gray-500">Grade</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 text-sm">Alice</td>
                  <td className="px-6 py-4 text-sm">1/01/1990</td>
                  <td className="px-6 py-4 text-sm">90</td>
                  <td className="px-6 py-4 text-sm">A</td>
                </tr>
                <tr>
                  <td className="px-6 py-4 text-sm">Bob</td>
                  <td className="px-6 py-4 text-sm">2/02/1991</td>
                  <td className="px-6 py-4 text-sm">75</td>
                  <td className="px-6 py-4 text-sm">B</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}