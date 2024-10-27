// src/App.js
import React from 'react';
import FraudForm from './components/FraudForm';
import Nav from './components/nav/Nav';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Banner from './components/Banner/Banner';
import './App.css';

function App() {
    return (
        <Router>
            <Nav /> {/* Move Nav outside of Routes for consistent display */}
            <Routes>
                <Route
                    path="/"
                    element={<Banner />} // Show Banner at root path
                />
                <Route
                    path="/fraud-form" // Use a specific path for FraudForm
                    element={<FraudForm />}
                />
            </Routes>
        </Router>
    );
}

export default App;
