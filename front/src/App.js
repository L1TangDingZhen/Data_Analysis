import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import './App.css';
import Data from './extra/Data';
function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <Routes>
            {/* 将 "/" 路径设置为 Data 组件 */}
            <Route path="/" element={<Data />} />
          </Routes>
        </header>
      </div>
    </Router>
  );
}

export default App;
