import React from 'react';
import "./banner.css";
import 'slick-carousel/slick/slick.css';
import 'slick-carousel/slick/slick-theme.css';


export default function Banner() {
  return (
    <div className='banner'>
      <div className="banner-item banner-item-02">
        <div className="text-content">
          <h4>Flash Deals</h4>
          <h2>Get your best products</h2>
        </div>
      </div>
      <div className="banner-item banner-item-03">
        <div className="text-content">
          <h4>Last Minute</h4>
          <h2>Grab last-minute deals</h2>
        </div>
      </div>
   
    </div>
  );
}


