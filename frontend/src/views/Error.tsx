import React from "react";
import {Typography} from "antd";

export default class Error extends React.Component<any, any> {

    render() {
        return (<div style={{ paddingLeft: 36, paddingTop: 24 }}>
            <Typography.Title style={{ fontWeight: 300 }} level={2}>Ingen adgang</Typography.Title>
            Find en anden funktion i menuen. <br/>
            Hvis dette er en fejl - så skal en fakultets- eller systemadminstrator give dig adgang.
        </div>)
    }

    componentDidMount() {

    }
}