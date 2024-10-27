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
        purchase_hour: '',
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
