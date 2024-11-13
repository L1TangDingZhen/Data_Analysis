import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
import './App.css';
import Data from './extra/Data';
import logo from './logo.svg';
import AA from './extra/BA';
function App() {
  return (
    <Router>
      <div className="App">
        <header className="App-header">
          <nav>
            <Link to="/A" style={{ margin: '0 10px' }}>Data Page</Link>
          </nav>
          <Routes>
            {/* 将 "/" 路径设置为 Data 组件 */}
            <Route path="/" element={<Data />} />
            <Route path="/A" element={<AA />} />

          </Routes>
        </header>
      </div>
    </Router>
  );
}

export default App;
