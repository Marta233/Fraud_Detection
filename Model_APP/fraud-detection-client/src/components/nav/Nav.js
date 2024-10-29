// src/components/Nav.js
import React, { useState } from "react";
import "./nav.css"; // Ensure your CSS is correctly linked
import "./assets/css/fontawesome.css"; // Ensure font awesome is correctly linked
import { Link } from 'react-router-dom';


export default function Nav() {
  const [isNavOpen, setIsNavOpen] = useState(false);
  const handleToggleClick = () => {
    setIsNavOpen(!isNavOpen);
  };

  return (
    <header>
      <nav className={`navbar navbar-expand-lg ${isNavOpen ? "active" : ""}`}>
        <div className="container">
          <a className="navbar-brand" href="index.html">
            <h2>
            Adey <em>Innovations</em>  
            </h2>
          </a>
          <button
            className="navbar-toggler"
            type="button"
            onClick={handleToggleClick}
            aria-controls="navbarResponsive"
            aria-expanded={isNavOpen}
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className={`collapse navbar-collapse ${isNavOpen ? "show" : ""}`} id="navbarResponsive">
            <ul className="navbar-nav ml-auto">
              <li className="nav-item active">
                <a className="nav-link" href="index.html">
                  Home
                  <span className="sr-only"></span>
                </a>
              </li>
              <li className="nav-item">
                           <a> <Link className="nav-link" to="/fraud-form">Fraud Cases</Link></a>
                        </li>
              <li className="nav-item">
                <a className="nav-link" href="about.html">
                  About Us
                </a>
              </li>
              <li className="nav-item">
                <a className="nav-link" href="contact.html">
                  Contact Us
                </a>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </header>
  );
}
