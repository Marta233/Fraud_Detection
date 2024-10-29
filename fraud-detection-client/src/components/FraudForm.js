import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';

const FraudForm = () => {
    const [formData, setFormData] = useState({
        purchase_time: '',
        signup_time: '',
        source: '',
        browser: '',
        sex: '',
        purchase_value: '',
        age: '',
        country: ''
    });
    
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            setResult(response.data.message);
        } catch (error) {
            console.error("There was an error!", error.response ? error.response.data : error);
            setResult("Error: Could not get prediction.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="background-image">
             <h2>Fraud Detection Form</h2>
            <div className="form-container">
               
                <form onSubmit={handleSubmit}>
                    <div className="form-group">
                        <label>Purchase Time</label>
                        <input
                            type="datetime-local"
                            name="purchase_time"
                            value={formData.purchase_time}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Signup Time</label>
                        <input
                            type="datetime-local"
                            name="signup_time"
                            value={formData.signup_time}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Source</label>
                        <input
                            type="text"
                            name="source"
                            value={formData.source}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Browser</label>
                        <input
                            type="text"
                            name="browser"
                            value={formData.browser}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Gender</label>
                        <input
                            type="text"
                            name="gender"
                            value={formData.gender}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Purchase Value</label>
                        <input
                            type="number"
                            name="purchase_value"
                            value={formData.purchase_value}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Age</label>
                        <input
                            type="number"
                            name="age"
                            value={formData.age}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <div className="form-group">
                        <label>Country</label>
                        <input
                            type="text"
                            name="country"
                            value={formData.country}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    <button type="submit" disabled={loading}>
                        {loading ? "Loading..." : "Submit"}
                    </button>
                </form>
                {result && <h3>{result}</h3>}
            </div>
        </div>
    );
};

export default FraudForm;

import React, { useState } from 'react';
import axios from 'axios';
import './styles.css';  // Import the CSS file

const FraudForm = () => {
    const [formData, setFormData] = useState({
        purchase_value: '',
        age: '',
        purchase_day_of_week: '',
        signup_day_of_week: '',
        signup_hour: '',
        purchase__hour: '',
        category_Ads: '',
        category_Direct: '',
        category_SEO: '',
        category_Chrome: '',
        category_FireFox: '',
        category_IE: '',
        category_Opera: '',
        category_Safari: '',
        category_F: '',
        category_M: '',
        Country_encoded: ''
    });
    
    const [result, setResult] = useState('');
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            setResult(response.data.message);
        } catch (error) {
            console.error("There was an error!", error.response ? error.response.data : error);
            setResult("Error: Could not get prediction.");
        } finally {
            setLoading(false);  // Reset loading state
        }
    };

    return (
        <div className="background-image">
            <div className="background-image1">
        <div className="form-container">
            <h2>Fraud Dtection Form</h2>
            <form onSubmit={handleSubmit}>
                {Object.keys(formData).map((key) => (
                    <div key={key} className="form-group">
                        <input
                            type="number"
                            name={key}
                            value={formData[key]}
                            onChange={handleChange}
                            placeholder={key.replace(/_/g, ' ')} // Using placeholder instead of label
                            required
                        />
                    </div>
                ))}
                <button type="submit" disabled={loading}>
                    {loading ? "Loading..." : "Submit"}
                </button>
            </form>
            {result && <h3>{result}</h3>}
        </div>
        </div>
        </div>
    );
};

export default FraudForm;
